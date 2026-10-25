"""Phase 10 - Add chat memories table and memory preference

Revision ID: 7a8b9c0d1e2f
Revises: 6f7a8b9c0d1e
Create Date: 2026-10-25 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7a8b9c0d1e2f'
down_revision = '6f7a8b9c0d1e'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Add chat_memory_enabled column to user_preferences
    with op.batch_alter_table('user_preferences', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('chat_memory_enabled', sa.Boolean(), nullable=False, server_default=sa.true())
        )

    # 2. Create chat_memories table
    op.create_table(
        'chat_memories',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('memory_text', sa.String(length=256), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('chat_memories', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_chat_memories_user_id'), ['user_id'], unique=False)


def downgrade():
    op.drop_table('chat_memories')
    with op.batch_alter_table('user_preferences', schema=None) as batch_op:
        batch_op.drop_column('chat_memory_enabled')
