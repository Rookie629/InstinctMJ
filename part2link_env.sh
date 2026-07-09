#!/usr/bin/env bash
# Part2Link generated asset paths for the current workspace.
# Usage: source part2link_env.sh

export INSTINCT_PART2LINK_DATASET_ROOT=/home/yangke/KY/InstinctLab_interact/datasets/interaction/output_npz_29dof_with_object
export INSTINCT_PART2LINK_ASSET_CACHE=/home/yangke/KY/InstinctLab_interact/datasets/interaction/output_npz_29dof_with_object/derived/part2link_assets
export INSTINCT_PART2LINK_COLLISION_CACHE=/home/yangke/KY/InstinctLab_interact/datasets/interaction/output_npz_29dof_with_object/derived/part2link_coacd_collisions
export SITTING_PART2LINK_ALPHA_VALUES=1.0,0.8,0.5,0.0

# Current collision cache is complete for these 18 chairs.
# Remove this filter after chair_51_alpha_0p00 and all chair_55 alpha collisions are generated.
export SITTING_PART2LINK_CHAIR_NAMES=chair_14,chair_15,chair_17,chair_18,chair_20,chair_22,chair_28,chair_30,chair_32,chair_33,chair_37,chair_39,chair_40,chair_41,chair_43,chair_44,chair_46,chair_48
