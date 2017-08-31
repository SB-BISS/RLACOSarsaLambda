import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='m3d-v0',
    entry_point='gym_m3d.envs:m3d_environment.m3d',
    max_episode_steps=1000,
    reward_threshold=-110.0,
)