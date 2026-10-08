"""Phase 9.1 - Add timezone to user_preferences

Revision ID: 6f7a8b9c0d1e
Revises: 5e6f7a8b9c0d
Create Date: 2026-09-13 13:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6f7a8b9c0d1e'
down_revision = '5e6f7a8b9c0d'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('user_preferences', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('timezone', sa.String(length=50), nullable=True, server_default='UTC')
        )


def downgrade():
    with op.batch_alter_table('user_preferences', schema=None) as batch_op:
        batch_op.drop_column('timezone')
