import json
from datetime import datetime
from http import HTTPStatus
from math import ceil
from pathlib import Path
from typing import Optional

import alembic.command
import alembic.config
import fastapi
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
from fastapi_utils.tasks import repeat_every
from loguru import logger
from oasst_backend.api.deps import api_auth, create_api_client
from oasst_backend.api.v1.api import api_router
from oasst_backend.api.v1.utils import prepare_conversation
from oasst_backend.cached_stats_repository import CachedStatsRepository
from oasst_backend.config import settings
from oasst_backend.database import engine
from oasst_backend.models import message_tree_state
from oasst_backend.prompt_repository import PromptRepository, UserRepository
from oasst_backend.task_repository import TaskRepository, delete_expired_tasks
from oasst_backend.tree_manager import TreeManager, halt_prompts_of_disabled_users
from oasst_backend.user_stats_repository import UserStatsRepository, UserStatsTimeFrame
from oasst_backend.utils.database_utils import CommitMode, managed_tx_function
from oasst_shared.exceptions import OasstError, OasstErrorCode
from oasst_shared.schemas import protocol as protocol_schema
from oasst_shared.utils import utcnow
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sqlmodel import Session
from starlette.middleware.cors import CORSMiddleware
from some_ai_library import AIModelManager, AIError  # New AI feature integration

app = fastapi.FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")
startup_time: datetime = utcnow()

# Constants
FAILED_UPDATE_STATS_MSG = "Failed to update leader board stats."

@app.exception_handler(OasstError)
async def oasst_exception_handler(request: fastapi.Request, ex: OasstError):
    logger.error(f"{request.method} {request.url} failed: {repr(ex)}")
    return fastapi.responses.JSONResponse(
        status_code=int(ex.http_status_code),
        content=protocol_schema.OasstErrorResponse(
            message=ex.message,
            error_code=OasstErrorCode(ex.error_code),
        ).dict(),
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: fastapi.Request, ex: Exception):
    logger.exception(f"{request.method} {request.url} failed [UNHANDLED]: {repr(ex)}")
    status = HTTPStatus.INTERNAL_SERVER_ERROR
    return fastapi.responses.JSONResponse(
        status_code=status.value, content={"message": status.name, "error_code": OasstErrorCode.GENERIC_ERROR}
    )

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def upgrade_alembic():
    logger.info("Attempting to upgrade alembic on startup")
    try:
        alembic_ini_path = Path(__file__).parent / "alembic.ini"
        alembic_cfg = alembic.config.Config(str(alembic_ini_path))
        alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URI)
        alembic.command.upgrade(alembic_cfg, "head")
        logger.info("Successfully upgraded alembic on startup")
    except Exception:
        logger.exception("Alembic upgrade failed on startup")

def create_official_web_api_client():
    with Session(engine) as session:
        try:
            api_auth(settings.OFFICIAL_WEB_API_KEY, db=session)
        except OasstError:
            logger.info("Creating official web API client")
            create_api_client(
                session=session,
                api_key=settings.OFFICIAL_WEB_API_KEY,
                description="The official web client for the OASST backend.",
                frontend_type="web",
                trusted=True,
            )

async def connect_redis():
    async def http_callback(request: fastapi.Request, response: fastapi.Response, pexpire: int):
        """Error callback function when too many requests"""
        expire = ceil(pexpire / 1000)
        raise OasstError(
            f"Too Many Requests. Retry After {expire} seconds.",
            OasstErrorCode.TOO_MANY_REQUESTS,
            HTTPStatus.TOO_MANY_REQUESTS,
        )

    try:
        redis_client = redis.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0", encoding="utf-8", decode_responses=True
        )
        logger.info(f"Connected to {redis_client=}")
        await FastAPILimiter.init(redis_client, http_callback=http_callback)
    except Exception:
        logger.exception("Failed to establish Redis connection")

def create_seed_data(session: Session):
    class DummyMessage(BaseModel):
        task_message_id: str
        user_message_id: str
        parent_message_id: Optional[str]
        text: str
        lang: Optional[str]
        role: str
        tree_state: Optional[message_tree_state.State]

    if not settings.OFFICIAL_WEB_API_KEY:
        raise ValueError("Cannot use seed data without OFFICIAL_WEB_API_KEY")

    try:
        logger.info("Seed data check began")

        api_client = api_auth(settings.OFFICIAL_WEB_API_KEY, db=session)
        dummy_user = protocol_schema.User(id="__dummy_user__", display_name="Dummy User", auth_method="local")

        ur = UserRepository(db=session, api_client=api_client)
        tr = TaskRepository(db=session, api_client=api_client, client_user=dummy_user, user_repository=ur)
        ur.update_user(tr.user_id, enabled=True, show_on_leaderboard=False, tos_acceptance=True)
        pr = PromptRepository(
            db=session, api_client=api_client, client_user=dummy_user, user_repository=ur, task_repository=tr
        )
        tm = TreeManager(session, pr)

        with open(settings.DEBUG_USE_SEED_DATA_PATH) as f:
            dummy_messages_raw = json.load(f)

        dummy_messages = [DummyMessage(**dm) for dm in dummy_messages_raw]

        for msg in dummy_messages:
            task = tr.fetch_task_by_frontend_message_id(msg.task_message_id)
            if task and not task.ack:
                logger.warning("Deleting unacknowledged seed data task")
                session.delete(task)
                task = None
            if not task:
                if msg.parent_message_id is None:
                    task = tr.store_task(
                        protocol_schema.InitialPromptTask(hint=""), message_tree_id=None, parent_message_id=None
                    )
                else:
                    parent_message = pr.fetch_message_by_frontend_message_id(
                        msg.parent_message_id, fail_if_missing=True
                    )
                    conversation_messages = pr.fetch_message_conversation(parent_message)
                    conversation = prepare_conversation(conversation_messages)
                    if msg.role == "assistant":
                        task = tr.store_task(
                            protocol_schema.AssistantReplyTask(conversation=conversation),
                            message_tree_id=parent_message.message_tree_id,
                            parent_message_id=parent_message.id,
                        )
                    else:
                        task = tr.store_task(
                            protocol_schema.PrompterReplyTask(conversation=conversation),
                            message_tree_id=parent_message.message_tree_id,
                            parent_message_id=parent_message.id,
                        )
                tr.bind_frontend_message_id(task.id, msg.task_message_id)
                message = pr.store_text_reply(
                    msg.text,
                    msg.lang or "en",
                    msg.task_message_id,
                    msg.user_message_id,
                    review_count=5,
                    review_result=True,
                    check_tree_state=False,
                    check_duplicate=False,
                )
                if message.parent_id is None:
                    tm._insert_default_state(
                        root_message_id=message.id,
                        lang=message.lang,
                        state=msg.tree_state or message_tree_state.State.GROWING,
                    )
                    session.flush()

                logger.info(
                    f"Inserted: message_id: {message.id}, payload: {message.payload.payload}, parent_message_id: {message.parent_id}"
                )
            else:
                logger.debug(f"seed data task found: {task.id}")

        logger.info("Seed data check completed")

    except Exception:
        logger.exception("Seed data insertion failed")

@app.on_event("startup")
def initialize_startup_tasks():
    if settings.UPDATE_ALEMBIC:
        upgrade_alembic()
    if settings.OFFICIAL_WEB_API_KEY:
        create_official_web_api_client()
    if settings.ENABLE_PROM_METRICS:
        Instrumentator().instrument(app).expose(app)
    if settings.RATE_LIMIT:
        app.on_event("startup")(connect_redis)
    if settings.DEBUG_USE_SEED_DATA:
        app.on_event("startup")(create_seed_data)
    app.on_event("startup")(ensure_tree_states)

@app.on_event("startup")
@repeat_every(seconds=60 * settings.USER_STATS_INTERVAL_DAY, wait_first=False)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def update_leader_board_day(session: Session) -> None:
    try:
        UserStatsRepository(db=session).update_stats(UserStatsTimeFrame.DAY)
        CachedStatsRepository(db=session).update_stats(UserStatsTimeFrame.DAY)
    except Exception:
        logger.exception(FAILED_UPDATE_STATS_MSG)

@app.on_event("startup")
@repeat_every(seconds=60 * settings.USER_STATS_INTERVAL_WEEK, wait_first=False)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def update_leader_board_week(session: Session) -> None:
    try:
        UserStatsRepository(db=session).update_stats(UserStatsTimeFrame.WEEK)
        CachedStatsRepository(db=session).update_stats(UserStatsTimeFrame.WEEK)
    except Exception:
        logger.exception(FAILED_UPDATE_STATS_MSG)

@app.on_event("startup")
@repeat_every(seconds=60 * settings.USER_STATS_INTERVAL_MONTH, wait_first=False)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def update_leader_board_month(session: Session) -> None:
    try:
        UserStatsRepository(db=session).update_stats(UserStatsTimeFrame.MONTH)
        CachedStatsRepository(db=session).update_stats(UserStatsTimeFrame.MONTH)
    except Exception:
        logger.exception(FAILED_UPDATE_STATS_MSG)

@app.on_event("startup")
@repeat_every(seconds=60 * settings.DELETE_EXPIRED_TASK_INTERVAL, wait_first=False)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def remove_expired_tasks(session: Session) -> None:
    try:
        delete_expired_tasks(session)
    except Exception:
        logger.exception("Failed to delete expired tasks.")

@app.on_event("startup")
@repeat_every(seconds=60 * settings.TASKS_WITHOUT_MESSAGES_INTERVAL, wait_first=False)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def delete_tasks_without_messages(session: Session) -> None:
    try:
        delete_expired_tasks(session)
    except Exception:
        logger.exception("Failed to delete tasks without messages.")

@app.on_event("startup")
@repeat_every(seconds=60 * settings.TASKS_STALE_INTERVAL, wait_first=False)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def delete_stale_tasks(session: Session) -> None:
    try:
        delete_expired_tasks(session)
    except Exception:
        logger.exception("Failed to delete stale tasks.")

@app.on_event("startup")
@repeat_every(seconds=60 * settings.PROMPTS_WITHOUT_MESSAGES_INTERVAL, wait_first=False)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def delete_prompts_without_messages(session: Session) -> None:
    try:
        delete_expired_tasks(session)
    except Exception:
        logger.exception("Failed to delete prompts without messages.")

@app.on_event("startup")
@repeat_every(seconds=60 * settings.PROMPTS_STALE_INTERVAL, wait_first=False)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def delete_stale_prompts(session: Session) -> None:
    try:
        delete_expired_tasks(session)
    except Exception:
        logger.exception("Failed to delete stale prompts.")

@app.on_event("startup")
@repeat_every(seconds=60 * settings.HALTING_PROMPTS_INTERVAL, wait_first=False)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def halt_prompts(session: Session) -> None:
    try:
        halt_prompts_of_disabled_users(session)
    except Exception:
        logger.exception("Failed to halt prompts of disabled users.")

# AI Model Manager Initialization
@app.on_event("startup")
async def initialize_ai_model_manager():
    try:
        ai_model_manager = AIModelManager()
        await ai_model_manager.initialize_models()
        logger.info("AI Model Manager initialized successfully.")
    except AIError:
        logger.exception("AI Model Manager initialization failed.")

app.include_router(api_router, prefix=settings.API_V1_STR)
