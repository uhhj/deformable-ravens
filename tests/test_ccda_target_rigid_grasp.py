from ravens import gripper
from ravens.gripper import Suction


TARGET_ID = 21
DISTRACTOR_ID = 8
SUCTION_ID = 30


def contact_point(
    object_id,
    contact_link=-1,
    normal_force=1.0,
):
    point = [0] * 10
    point[2] = int(object_id)
    point[4] = int(contact_link)
    point[9] = float(normal_force)
    return tuple(point)


def make_suction():
    suction = object.__new__(
        Suction
    )
    suction.body = SUCTION_ID
    suction.activated = False
    suction.contact_constraint = None
    suction.def_grip_item = None
    suction.def_grip_anchors = None
    suction.def_min_vertex = None
    suction.def_min_vetex = None
    suction.def_min_distance = None
    suction.init_grip_distance = None
    suction.init_grip_item = None
    return suction


def install_pybullet(
    monkeypatch,
    *,
    include_target,
    include_distractor,
):
    points = []
    if include_target:
        points.append(
            contact_point(TARGET_ID)
        )
    if include_distractor:
        points.append(
            contact_point(
                DISTRACTOR_ID
            )
        )

    def get_contact_points(
        bodyA,
        linkIndexA,
        bodyB=None,
    ):
        assert bodyA == SUCTION_ID
        assert linkIndexA == 0
        if bodyB is None:
            return tuple(points)
        return tuple(
            point
            for point in points
            if int(point[2])
            == int(bodyB)
        )

    created = []

    monkeypatch.setattr(
        gripper.p,
        "getContactPoints",
        get_contact_points,
    )
    monkeypatch.setattr(
        gripper.p,
        "getLinkState",
        lambda *args, **kwargs: (
            (
                0.4,
                -0.3,
                0.01,
            ),
            (
                0.0,
                0.0,
                0.0,
                1.0,
            ),
        ),
    )
    monkeypatch.setattr(
        gripper.p,
        "getBasePositionAndOrientation",
        lambda object_id: (
            (
                0.4,
                -0.3,
                0.005,
            ),
            (
                0.0,
                0.0,
                0.0,
                1.0,
            ),
        ),
    )
    monkeypatch.setattr(
        gripper.p,
        "invertTransform",
        lambda position, rotation: (
            position,
            rotation,
        ),
    )
    monkeypatch.setattr(
        gripper.p,
        "multiplyTransforms",
        lambda *args: (
            (
                0.0,
                0.0,
                0.0,
            ),
            (
                0.0,
                0.0,
                0.0,
                1.0,
            ),
        ),
    )

    def create_constraint(**kwargs):
        created.append(dict(kwargs))
        return 91

    monkeypatch.setattr(
        gripper.p,
        "createConstraint",
        create_constraint,
    )
    return created


def test_target_contact_ignores_distractor(
    monkeypatch,
):
    suction = make_suction()
    install_pybullet(
        monkeypatch,
        include_target=False,
        include_distractor=True,
    )

    assert not suction.detect_target_contact(
        TARGET_ID
    )


def test_target_activation_selects_target(
    monkeypatch,
):
    suction = make_suction()
    created = install_pybullet(
        monkeypatch,
        include_target=True,
        include_distractor=True,
    )

    suction.activate(
        possible_objects=[
            TARGET_ID,
            DISTRACTOR_ID,
        ],
        def_IDs=[],
        target_object_id=TARGET_ID,
    )

    assert suction.activated
    assert suction.contact_constraint == 91
    assert suction.init_grip_item == (
        TARGET_ID
    )
    assert len(created) == 1
    assert created[0][
        "childBodyUniqueId"
    ] == TARGET_ID


def test_target_activation_does_not_fall_back(
    monkeypatch,
):
    suction = make_suction()
    created = install_pybullet(
        monkeypatch,
        include_target=False,
        include_distractor=True,
    )

    suction.activate(
        possible_objects=[
            TARGET_ID,
            DISTRACTOR_ID,
        ],
        def_IDs=[],
        target_object_id=TARGET_ID,
    )

    assert suction.activated
    assert (
        suction.contact_constraint
        is None
    )
    assert created == []
