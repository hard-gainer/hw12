import uuid
import asyncio
import csv
import secrets
import json
from typing import Any, List, Optional, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, Integer, select, func, distinct, delete, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import UUID

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Query,
    Header,
    APIRouter,
    BackgroundTasks,
)
from pydantic import BaseModel, Field
from passlib.context import CryptContext
import redis.asyncio as redis


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    user_token = Column(String, unique=True, nullable=True, index=True)

    def __repr__(self) -> str:
        return f"User(username={self.username})"


class UserRegister(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    uuid: str
    username: str

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    user_token: str
    username: str


class StudentCreate(BaseModel):
    surname: str
    name: str
    faculty: str
    grade: str
    mark: int = Field(ge=0, le=100)


class StudentUpdate(BaseModel):
    surname: Optional[str] = None
    name: Optional[str] = None
    faculty: Optional[str] = None
    grade: Optional[str] = None
    mark: Optional[int] = Field(None, ge=0, le=100)


class StudentResponse(BaseModel):
    uuid: str
    surname: str
    name: str
    faculty: str
    grade: str
    mark: int

    class Config:
        from_attributes = True


class BulkDeleteRequest(BaseModel):
    student_uuids: List[str] = Field(..., description="List of student UUIDs to delete")


class CSVUploadRequest(BaseModel):
    file_path: str = Field(..., description="Path to CSV file to upload")


class AverageMarkResponse(BaseModel):
    faculty: str
    average_mark: float


class Student(Base):
    __tablename__ = "students"

    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    surname = Column(String, nullable=False)
    name = Column(String, nullable=False)
    faculty = Column(String, nullable=False)
    grade = Column(String, nullable=False)
    mark = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        return f"Student(surname={self.surname}, name={self.name}, faculty={self.faculty}, grade={self.grade}, mark={self.mark})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuid": str(self.uuid),
            "surname": self.surname,
            "name": self.name,
            "faculty": self.faculty,
            "grade": self.grade,
            "mark": self.mark,
        }


class StudentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def insert(self, student: Student) -> Student:
        """Создание студента"""
        self.session.add(student)
        await self.session.commit()
        await self.session.refresh(student)
        return student

    async def insert_many(self, students: List[Student]) -> None:
        """Массовое создание студентов"""
        self.session.add_all(students)
        await self.session.commit()

    async def select_all(self) -> List[Student]:
        """Получение всех студентов"""
        result = await self.session.execute(select(Student))
        return list(result.scalars().all())

    async def select_by_uuid(self, student_uuid: uuid.UUID) -> Optional[Student]:
        """Получение студента по UUID"""
        result = await self.session.execute(
            select(Student).where(Student.uuid == student_uuid)
        )
        return result.scalar_one_or_none()

    async def update_student(
        self, student_uuid: uuid.UUID, update_data: dict
    ) -> Optional[Student]:
        """Обновление данных студента"""
        update_data = {k: v for k, v in update_data.items() if v is not None}

        if not update_data:
            return await self.select_by_uuid(student_uuid)

        await self.session.execute(
            update(Student).where(Student.uuid == student_uuid).values(**update_data)
        )
        await self.session.commit()

        return await self.select_by_uuid(student_uuid)

    async def delete_student(self, student_uuid: uuid.UUID) -> bool:
        """Удаление студента"""
        result = await self.session.execute(
            delete(Student).where(Student.uuid == student_uuid)
        )
        await self.session.commit()
        return result.rowcount > 0

    async def bulk_delete_students(self, student_uuids: List[uuid.UUID]) -> int:
        """Массовое удаление студентов"""
        result = await self.session.execute(
            delete(Student).where(Student.uuid.in_(student_uuids))
        )
        await self.session.commit()
        return result.rowcount

    async def select_by_surname(self, surname: str) -> List[Student]:
        """Получение студентов по фамилии"""
        result = await self.session.execute(
            select(Student).where(Student.surname == surname)
        )
        return list(result.scalars().all())

    async def load_from_csv(self, filepath: str) -> None:
        """Загрузка данных из CSV"""
        students = []

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                student = Student(
                    surname=row["Фамилия"],
                    name=row["Имя"],
                    faculty=row["Факультет"],
                    grade=row["Курс"],
                    mark=int(row["Оценка"]),
                )
                students.append(student)

        await self.insert_many(students)

    async def get_students_by_faculty(self, faculty: str) -> List[Student]:
        """Получение студентов по факультету"""
        result = await self.session.execute(
            select(Student).where(Student.faculty == faculty)
        )
        return list(result.scalars().all())

    async def get_unique_grades(self) -> List[str]:
        """Получение уникальных курсов"""
        result = await self.session.execute(select(distinct(Student.grade)))
        return list(result.scalars().all())

    async def get_average_mark_by_faculty(self, faculty: str) -> Optional[float]:
        """Получение среднего балла по факультету"""
        result = await self.session.execute(
            select(func.avg(Student.mark)).where(Student.faculty == faculty)
        )
        avg_mark = result.scalar_one_or_none()
        return float(avg_mark) if avg_mark is not None else None

    async def get_students_by_grade_with_low_marks(
        self, grade: str, threshold: int = 30
    ) -> List[Student]:
        """Получение студентов с низкими оценками по курсу"""
        result = await self.session.execute(
            select(Student)
            .where(Student.grade == grade)
            .where(Student.mark < threshold)
        )
        return list(result.scalars().all())


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_user(self, username: str, password: str) -> User:
        """Создание нового пользователя"""
        hashed_password = pwd_context.hash(password)
        user = User(username=username, hashed_password=hashed_password)
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Получение пользователя по имени"""
        result = await self.session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_user_by_token(self, token: str) -> Optional[User]:
        """Получение пользователя по токену"""
        result = await self.session.execute(
            select(User).where(User.user_token == token)
        )
        return result.scalar_one_or_none()

    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Проверка пароля"""
        return pwd_context.verify(plain_password, hashed_password)

    async def generate_token(self, user: User) -> str:
        """Генерация и сохранение токена для пользователя"""
        token = secrets.token_urlsafe(32)
        user.user_token = token
        await self.session.commit()
        await self.session.refresh(user)
        return token

    async def revoke_token(self, user: User) -> None:
        """Отзыв токена пользователя (выход из системы)"""
        user.user_token = None
        await self.session.commit()


engine = None
async_session_maker = None
redis_client = None


async def init_db():
    """Инициализация БД"""
    global engine, async_session_maker

    engine = create_async_engine(
        "postgresql+asyncpg://user:password@localhost:5432/postgres", echo=True
    )

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def init_redis():
    """Инициализация Redis"""
    global redis_client
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


async def get_redis() -> redis.Redis:
    """Dependency для получения Redis клиента"""
    return redis_client


async def get_cache(key: str) -> Optional[str]:
    """Получить значение из кеша"""
    if redis_client:
        try:
            return await redis_client.get(key)
        except Exception as e:
            print(f"Redis get error: {e}")
    return None


async def set_cache(key: str, value: str, expire: int = 300) -> None:
    """Сохранить значение в кеш"""
    if redis_client:
        try:
            await redis_client.setex(key, expire, value)
        except Exception as e:
            print(f"Redis set error: {e}")


async def invalidate_cache_pattern(pattern: str) -> None:
    """Инвалидировать кеш по шаблону"""
    if redis_client:
        try:
            keys = await redis_client.keys(pattern)
            if keys:
                await redis_client.delete(*keys)
        except Exception as e:
            print(f"Redis invalidate error: {e}")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency для получения сессии БД"""
    async with async_session_maker() as session:
        yield session


async def get_current_user(
    authorization: str = Header(..., alias="Authorization"),
    session: AsyncSession = Depends(get_session),
) -> User:
    """Dependency для проверки аутентификации по токену"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.replace("Bearer ", "")

    repo = UserRepository(session)
    user = await repo.get_user_by_token(token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle для FastAPI"""
    await init_db()
    await init_redis()
    yield
    if engine:
        await engine.dispose()
    if redis_client:
        await redis_client.close()


app = FastAPI(lifespan=lifespan)

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


@auth_router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    user_data: UserRegister, session: AsyncSession = Depends(get_session)
):
    """Регистрация нового пользователя"""
    repo = UserRepository(session)

    existing_user = await repo.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    user = await repo.create_user(user_data.username, user_data.password)
    return UserResponse(uuid=str(user.uuid), username=user.username)


@auth_router.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin, session: AsyncSession = Depends(get_session)):
    """Вход пользователя (аутентификация)"""
    repo = UserRepository(session)

    user = await repo.get_user_by_username(user_data.username)
    if not user or not await repo.verify_password(
        user_data.password, user.hashed_password
    ):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = await repo.generate_token(user)
    return TokenResponse(user_token=token, username=user.username)


@auth_router.post("/logout", status_code=204)
async def logout(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Выход пользователя (завершение сессии)"""
    repo = UserRepository(session)
    await repo.revoke_token(current_user)
    return None


app.include_router(auth_router)


async def background_load_csv(file_path: str):
    """Фоновая задача загрузки данных из CSV"""
    try:
        async with async_session_maker() as session:
            repo = StudentRepository(session)
            await repo.load_from_csv(file_path)
            # Инвалидировать кеш после загрузки
            await invalidate_cache_pattern("students:*")
        print(f"CSV file {file_path} loaded successfully")
    except Exception as e:
        print(f"Error loading CSV: {e}")


async def background_bulk_delete(student_uuids: List[str]):
    """Фоновая задача массового удаления студентов"""
    try:
        async with async_session_maker() as session:
            repo = StudentRepository(session)
            uuids = [uuid.UUID(uid) for uid in student_uuids]
            deleted_count = await repo.bulk_delete_students(uuids)
            # Инвалидировать кеш после удаления
            await invalidate_cache_pattern("students:*")
        print(f"Deleted {deleted_count} students")
    except Exception as e:
        print(f"Error deleting students: {e}")


@app.post("/students/upload-csv", status_code=202)
async def upload_csv(
    request: CSVUploadRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """Загрузка данных из CSV файла в фоновом режиме"""
    background_tasks.add_task(background_load_csv, request.file_path)
    return {
        "message": "CSV upload started in background",
        "file_path": request.file_path,
    }


@app.delete("/students/bulk", status_code=202)
async def bulk_delete_students(
    request: BulkDeleteRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """Массовое удаление студентов в фоновом режиме"""
    background_tasks.add_task(background_bulk_delete, request.student_uuids)
    return {
        "message": "Bulk delete started in background",
        "count": len(request.student_uuids),
    }


@app.post("/students/", response_model=StudentResponse, status_code=201)
async def create_student(
    student_data: StudentCreate,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Создание нового студента"""
    repo = StudentRepository(session)

    student = Student(
        surname=student_data.surname,
        name=student_data.name,
        faculty=student_data.faculty,
        grade=student_data.grade,
        mark=student_data.mark,
    )

    created_student = await repo.insert(student)
    await invalidate_cache_pattern("students:*")
    return StudentResponse.model_validate(created_student)


@app.get("/students/", response_model=List[StudentResponse])
async def get_all_students(
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Получение всех студентов"""
    cache_key = "students:all"

    cached = await get_cache(cache_key)
    if cached:
        return json.loads(cached)

    repo = StudentRepository(session)
    students = await repo.select_all()
    result = [StudentResponse.model_validate(s).model_dump() for s in students]

    await set_cache(cache_key, json.dumps(result), expire=300)

    return result


@app.get("/students/{student_uuid}", response_model=StudentResponse)
async def get_student(
    student_uuid: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Получение студента по UUID"""
    cache_key = f"students:uuid:{student_uuid}"

    cached = await get_cache(cache_key)
    if cached:
        return json.loads(cached)

    repo = StudentRepository(session)
    student = await repo.select_by_uuid(student_uuid)

    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    result = StudentResponse.model_validate(student).model_dump()

    await set_cache(cache_key, json.dumps(result), expire=300)

    return result


@app.put("/students/{student_uuid}", response_model=StudentResponse)
async def update_student(
    student_uuid: uuid.UUID,
    student_data: StudentUpdate,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Обновление данных студента"""
    repo = StudentRepository(session)

    existing_student = await repo.select_by_uuid(student_uuid)
    if not existing_student:
        raise HTTPException(status_code=404, detail="Student not found")

    updated_student = await repo.update_student(
        student_uuid, student_data.model_dump(exclude_unset=True)
    )

    await invalidate_cache_pattern("students:*")

    return StudentResponse.model_validate(updated_student)


@app.delete("/students/{student_uuid}", status_code=204)
async def delete_student(
    student_uuid: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Удаление студента"""
    repo = StudentRepository(session)

    deleted = await repo.delete_student(student_uuid)

    if not deleted:
        raise HTTPException(status_code=404, detail="Student not found")

    await invalidate_cache_pattern("students:*")

    return None


async def main():
    await init_db()

    async with async_session_maker() as session:
        repo = StudentRepository(session)

        await repo.load_from_csv("students.csv")

        all_students = await repo.select_all()
        print(f"Всего студентов: {len(all_students)}\n")

        fpmі_students = await repo.get_students_by_faculty("ФПМИ")
        print(f"Студентов на ФПМИ: {len(fpmі_students)}")

        unique_grades = await repo.get_unique_grades()
        print(f"Уникальные курсы: {unique_grades}\n")

        avg_mark = await repo.get_average_mark_by_faculty("ФПМИ")
        print(f"Средний балл на ФПМИ: {avg_mark:.2f}\n")


if __name__ == "__main__":
    asyncio.run(main())
