"""The graph engine: it follows declared edges, refuses undeclared ones, and bounds cycles."""

import pytest

from app.agents.graph import END, Graph, GraphError, Next, Step, StepBudgetExceeded


def graph(nodes, edges, **kwargs):
    return Graph(nodes=nodes, edges=edges, entry=next(iter(nodes)), **kwargs)


async def test_it_walks_the_graph_and_reports_every_transition():
    async def first(state):
        state.append("first")
        return Next("second", note="hello")

    async def second(state):
        state.append("second")
        return Next(END)

    g = graph({"first": first, "second": second}, {"first": ("second",), "second": (END,)})
    visited: list[Step] = []
    steps = await g.run(state := [], on_step=visited.append)

    assert state == ["first", "second"]
    assert [(s.node, s.next, s.note) for s in steps] == [
        ("first", "second", "hello"),
        ("second", END, None),
    ]
    assert visited == steps  # on_step sees them as they happen, for tracing
    assert all(s.duration_ms >= 0 for s in steps)


async def test_a_node_cannot_jump_somewhere_its_edges_dont_allow():
    # The guard that stops a prompt (or a hallucinated step name) steering the run off the graph.
    async def rogue(_state):
        return Next("somewhere_else")

    g = graph({"rogue": rogue}, {"rogue": (END,)})
    with pytest.raises(GraphError, match="not one of"):
        await g.run([])


async def test_a_cycle_stops_at_the_step_budget():
    async def loop(_state):
        return Next("loop")

    g = graph({"loop": loop}, {"loop": ("loop", END)}, max_steps=5)
    with pytest.raises(StepBudgetExceeded, match="within 5 steps"):
        await g.run([])


@pytest.mark.parametrize(
    ("nodes", "edges", "message"),
    [
        ({"a": None}, {"a": ("b",)}, "undefined node"),
        ({"a": None}, {"a": ()}, "no outgoing edges"),
        ({"a": None, "b": None}, {"a": (END,)}, "declared edges"),
    ],
)
def test_a_malformed_graph_is_rejected_when_it_is_built(nodes, edges, message):
    # Not at run time, halfway through a user's request.
    with pytest.raises(GraphError, match=message):
        graph(nodes, edges)


def test_an_entry_node_that_doesnt_exist_is_rejected():
    with pytest.raises(GraphError, match="entry node 'nowhere' is not defined"):
        Graph(nodes={"a": None}, edges={"a": (END,)}, entry="nowhere")


def test_the_diagram_is_generated_from_the_edges():
    g = graph(
        {"a": None, "b": None},
        {"a": ("b", END), "b": ("a",)},
    )
    mermaid = g.to_mermaid()
    assert "start([start]) --> a" in mermaid
    assert "a --> b" in mermaid and "a --> done([done])" in mermaid
    assert "b --> a" in mermaid  # the cycle is drawn, not hidden
