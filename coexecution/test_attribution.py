"""Accounting checks for CPU launch attribution, without running a model."""

from coexecution.operator_analysis import assign_launches, operator_family


def test_launch_uses_innermost_range_on_its_own_thread():
    ranges = [
        (0, 100, 1, "S1/serial/input=1/cache=0"),
        (5, 90, 1, "MODULE/S1/dit"),
        (10, 30, 1, "MODULE/S1/dit.linear"),
        (11, 25, 1, "OP/aten.mm.default/0"),
        (0, 100, 2, "S2/serial/input=1/publish=1"),
        (10, 90, 2, "MODULE/S2/vision"),
    ]
    launches = [(7, 15, 1), (8, 40, 1), (9, 15, 2), (10, 101, 1)]
    result = assign_launches(ranges, launches)
    assert result[7] == {
        "worker": "S1/serial/input=1/cache=0",
        "module": "MODULE/S1/dit.linear",
        "operator": "OP/aten.mm.default/0",
    }
    assert result[8]["module"] == "MODULE/S1/dit"
    assert result[8]["operator"] is None
    assert result[9]["module"] == "MODULE/S2/vision"
    assert result[10]["worker"] is None


def test_launch_end_boundary_is_exclusive():
    result = assign_launches([(5, 10, 1, "OP/a/0")], [(1, 5, 1), (2, 10, 1)])
    assert result[1]["operator"] == "OP/a/0"
    assert result[2]["operator"] is None


def test_real_high_level_operator_categories():
    assert operator_family("aten.linear.default") == "Linear / matrix multiplication"
    assert operator_family("aten.conv3d.default") == "Convolution"
    assert operator_family("aten.to.dtype") == "Dtype / device conversion"
    assert (
        operator_family("aten.scaled_dot_product_attention.default") == "SDPA attention"
    )
