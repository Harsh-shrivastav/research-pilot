"""Shared dependencies. Phase 1 has no auth flow yet, so we resolve a single
seeded dev user. This is the seam where real auth will plug in later —
everything downstream already keys off owner_id."""
from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User

DEV_USER_EMAIL = "dev@researchpilot.local"


def get_current_user(db: Session = Depends(get_db)) -> User:
    user = db.scalar(select(User).where(User.email == DEV_USER_EMAIL))
    if user is None:
        user = User(email=DEV_USER_EMAIL, name="Dev Researcher")
        db.add(user)
        db.commit()
        db.refresh(user)
    return user
