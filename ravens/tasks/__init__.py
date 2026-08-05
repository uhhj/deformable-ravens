from ravens.tasks.task import Task
from ravens.tasks.sorting import Sorting
from ravens.tasks.insertion import Insertion
from ravens.tasks.insertion import InsertionTranslation
from ravens.tasks.hanoi import Hanoi
from ravens.tasks.aligning import Aligning
from ravens.tasks.stacking import Stacking
from ravens.tasks.sweeping import Sweeping
from ravens.tasks.pushing import Pushing
from ravens.tasks.palletizing import Palletizing
from ravens.tasks.kitting import Kitting
from ravens.tasks.packing import Packing
from ravens.tasks.cable import Cable

# New customized environments. When adding these envs, double check:
#   Environment._is_new_cable_env()
#   Environment._is_cloth_env()
#   Environment._is_bag_env()
# and adjust those methods as needed.

from ravens.tasks.insertion_goal import InsertionGoal
from ravens.tasks.defs_cables import (
        CableShape, CableShapeNoTarget, CableLineNoTarget,
        CableRing, CableRingNoTarget)
from ravens.tasks.defs_cloth import (
        ClothFlat, ClothFlatNoTarget, ClothCover)
from ravens.tasks.ccda_slack_cable_v2 import CCDASlackCableV2
from ravens.tasks.ccda_hidden_friction_cable import CCDAHiddenFrictionCable
from ravens.tasks.ccda_hidden_hook_cable import CCDAHiddenHookCable
from ravens.tasks.ccda_hidden_latch_cable import CCDAHiddenLatchCable
from ravens.tasks.defs_bags import (
        BagAloneOpen, BagItemsEasy, BagItemsHard, BagColorGoal)

names = {'sorting':             Sorting,
         'insertion':           Insertion,
         'insertion-translation': InsertionTranslation,
         'hanoi':               Hanoi,
         'aligning':            Aligning,
         'stacking':            Stacking,
         'sweeping':            Sweeping,
         'pushing':             Pushing,
         'palletizing':         Palletizing,
         'kitting':             Kitting,
         'packing':             Packing,
         'cable':               Cable,
         'insertion-goal':      InsertionGoal, # start of custom envs
         'cable-shape':         CableShape,
         'cable-shape-notarget': CableShapeNoTarget,
         'cable-line-notarget': CableLineNoTarget,
         'ccda-slack-cable-v2': CCDASlackCableV2,
         'ccda-hidden-friction-cable': CCDAHiddenFrictionCable,
         'ccda-hidden-hook-cable': CCDAHiddenHookCable,
         'ccda-hidden-latch-cable': CCDAHiddenLatchCable,
         'cable-ring':          CableRing,
         'cable-ring-notarget': CableRingNoTarget,
         'cloth-flat':          ClothFlat,
         'cloth-flat-notarget': ClothFlatNoTarget,
         'cloth-cover':         ClothCover,
         'bag-alone-open':      BagAloneOpen,
         'bag-items-easy':      BagItemsEasy,
         'bag-items-hard':      BagItemsHard,
         'bag-color-goal':      BagColorGoal,
}
