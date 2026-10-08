"""Phase 7.1 - Add partial unique index uix_task_user_activity_active to wellness_tasks

Revision ID: 5e6f7a8b9c0d
Revises: 4d5e6f7a8b9c
Create Date: 2026-09-13 11:46:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5e6f7a8b9c0d'
down_revision = '4d5e6f7a8b9c'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Resolve pre-existing duplicate active tasks safely without deleting student history
    connection = op.get_bind()
    connection.execute(sa.text("""
        UPDATE wellness_tasks
        SET completed = 1, completed_at = created_at
        WHERE id IN (
            SELECT t1.id
            FROM wellness_tasks t1
            JOIN wellness_tasks t2 ON t1.user_id = t2.user_id 
                                 AND t1.activity_id = t2.activity_id 
                                 AND t1.completed = 0 
                                 AND t2.completed = 0
                                 AND t1.id > t2.id
            WHERE t1.activity_id IS NOT NULL
        )
    """))

    # 2. Create partial unique index enforcing at most 1 active task per (user_id, activity_id)
    with op.batch_alter_table('wellness_tasks', schema=None) as batch_op:
        batch_op.create_index(
            'uix_task_user_activity_active',
            ['user_id', 'activity_id'],
            unique=True,
            sqlite_where=sa.text('completed = 0 AND activity_id IS NOT NULL'),
            postgresql_where=sa.text('completed = false AND activity_id IS NOT NULL')
        )


def downgrade():
    with op.batch_alter_table('wellness_tasks', schema=None) as batch_op:
        batch_op.drop_index('uix_task_user_activity_active')
