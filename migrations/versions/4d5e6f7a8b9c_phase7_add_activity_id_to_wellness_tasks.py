"""Phase 7 - Add activity_id and ix_task_user_activity index to wellness_tasks

Revision ID: 4d5e6f7a8b9c
Revises: 3c4d4685691f
Create Date: 2026-09-13 11:40:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4d5e6f7a8b9c'
down_revision = '3c4d4685691f'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('wellness_tasks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('activity_id', sa.String(length=50), nullable=True))
        batch_op.create_index('ix_task_user_activity', ['user_id', 'activity_id'], unique=False)


def downgrade():
    with op.batch_alter_table('wellness_tasks', schema=None) as batch_op:
        batch_op.drop_index('ix_task_user_activity')
        batch_op.drop_column('activity_id')
