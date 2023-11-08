import sys

import numpy as np
from spot_rl.envs.nav_env import construct_config_for_nav
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.heuristic_nav import navigate_to_aria_goal
from spot_rl.utils.utils import get_default_parser, map_user_input_to_boolean


def parse_arguments(args=sys.argv[1:]):
    parser = get_default_parser()
    parser.add_argument(
        "-g",
        "--goal",
        help="input:string -> goal x,y,theta in meters and radians obtained from aria",
        required=True,
        type=str,
    )

    args = parser.parse_args(args=args)

    return args


if __name__ == "__main__":
    # args = parse_arguments()
    # assert args.goal is not None, "Please provide x,y,theta in meters & radians that you get from aria"
    # goal_str = str(args.goal).strip().split(",")
    # assert len(goal_str) == 3, f"Goal str len was supposed to be 3 but found {len(goal_str)}"
    # x, y, theta = [float(s) for s in goal_str]
    # 4.069178561212342, -3.8163447712013268, -1.9480954162660047
    x, y, theta = (
        3.97,
        -3.65,
        -1.1429722791562886,
    )  # , np.deg2rad(-96.2389683367895)
    print(f"Original Nav Goal {x, y, np.degrees(theta)}")
    # x, y = push_forward_point_along_theta_by_offset(x, y, theta, 0.3)
    # print(f"Nav goal after pushing forward by 0.3m {x,y}")
    nav_config = construct_config_for_nav()
    # nav_config.SUCCESS_DISTANCE = 0.20
    # nav_config.SUCCESS_ANGLE_DIST = 3
    nav_config.MAX_EPISODE_STEPS = 50
    spotskillmanager = SpotSkillManager(nav_config)
    object_target = "ball"
    print(
        f"Spot was able to reach the goal ? {navigate_to_aria_goal(x, y, theta, spotskillmanager, object_target=object_target, pull_back=False)}"
    )
    # spotskillmanager.pick("ball")
    should_dock = map_user_input_to_boolean("Do you want to dock & exit ?")
    if should_dock:
        spotskillmanager.nav_controller.nav_env.disable_nav_goal_change()
        spotskillmanager.dock()
