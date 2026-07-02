# Skill 5: LangGraph Orchestration — The Real Kitchen Opens

> You built the food counter (Skills 1–3). You ran a chaotic kitchen by hand (Skill 4).
> Now you install the proper ordering system.

---

## What's About to Happen

We're going to take your **exact** manual agent loop (`manual_agent_loop.py`) and rebuild it
as a LangGraph state machine. Same behavior. 1/4 the code. 10x more maintainable.

By the end, you'll have a graph that looks like this:

```
START → classify_node → [route] → tool_node     → respond_node → END
                                → rag_node      → respond_node → END
                                → escalate_node → respond_node → END
                                → direct_node   → respond_node → END
```

Every line of code will make architectural sense to you.

---

## Before We Write a Single Line: The 3 Mental Shifts

### Shift 1: From "flow control" to "graph declaration"

In your manual loop, the **code order IS the flow**:

```python
# manual_agent_loop.py — the flow is invisible
classification = step_classify(...)     # always first
route = step_route(classification, ...) # always second
if route == "tool":                     # branching buried in if/elif
    response = step_tool(...)
elif route == "rag":
    response = step_rag(...)
```

In LangGraph, you **declare the graph separately from the logic**:

```python
# The flow is VISIBLE — you can read it like a map
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", router, {
    "tool":     "tool_node",
    "rag":      "rag_node",
    "escalate": "escalate_node",
    "direct":   "direct_node",
})
```

**Why this matters:** When you add a 5th route next month, you add ONE line to the
edge declaration. In the manual loop, you add another `elif`, update the state handling,
and pray you didn't break the other routes.

---

### Shift 2: From "mutable global state" to "typed state with declared writes"

Your manual loop:

```python
# manual_agent_loop.py — who changed what? grep to find out
state["last_intent"] = result.intent.value       # line 106
state["last_tool_called"] = result["tool_called"] # line 167 (different function!)
state["last_rag_chunks"] = rag_result.sources     # line 189 (yet another function!)
```

LangGraph:

```python
# Each node DECLARES what it writes by returning a partial dict
def classify_node(state):
    ...
    return {"intent": "refund_request", "urgency": "high"}  # that's ALL this node writes

def tool_node(state):
    ...
    return {"tool_called": "order_lookup", "tool_result": {...}}  # that's ALL this node writes
```

**Why this matters:** When your agent produces a wrong response, you look at the node that
wrote the bad field. Not 400 lines of code — ONE function.

---

### Shift 3: From "one big function" to "composable nodes"

In your manual loop, `run_agent_loop()` is 150+ lines. It does EVERYTHING — init clients,
get input, classify, route, act, respond, handle errors.

In LangGraph, each node is a **small, testable function** that does exactly ONE thing:

```python
def classify_node(state) -> dict:    # 10 lines. Testable alone.
def tool_node(state) -> dict:        # 15 lines. Testable alone.
def rag_node(state) -> dict:         # 12 lines. Testable alone.
def respond_node(state) -> dict:     # 8 lines. Testable alone.
```

You can test `classify_node` without running the whole graph.
You can swap `rag_node` for a better version without touching anything else.

---

## Part 1: Setting Up — What Gets Installed and Why

```bash
pip install langgraph langchain-core
```

**Wait — why `langchain-core`?**

LangGraph is technically **separate** from LangChain. But it uses `langchain-core` for:
- Message types (`HumanMessage`, `AIMessage`)  — optional, we won't use these yet
- Some base abstractions

We're NOT using LangChain chains, agents, or memory. Just the graph engine.

> Think of it like installing a car engine (LangGraph) that happens to use
> standard bolts from one specific manufacturer (langchain-core).
> You don't need the manufacturer's whole car — just the bolts.

---

## Part 2: AgentState — Designing the Order Ticket

This is the FIRST thing you design. Before any node. Before any edge.

**Why?** Because state is the CONTRACT between all nodes. Every node reads from it
and writes to it. If the state is wrong, everything downstream breaks.

```python
from typing import TypedDict, Optional

class AgentState(TypedDict):
    """The order ticket that travels through every station in our kitchen.
    
    DESIGN RULE: Every field here must answer one of two questions:
      1. "What do we know about the customer's request?" (input data)
      2. "What has the agent done so far?" (processing data)
    
    If a field doesn't answer either question, it doesn't belong here.
    """
    
    # ── What we know about the request ──────────────────────
    user_message: str                    # The raw customer message
    intent: str                          # What they want (from classify_node)
    urgency: str                         # How urgent (from classify_node)
    sentiment: str                       # How they feel (from classify_node)
    confidence: float                    # How sure are we (from classify_node)
    
    # ── What the agent has done ─────────────────────────────
    route: str                           # Which path we took (from router)
    tool_called: Optional[str]           # Which tool was called (from tool_node)
    tool_result: Optional[dict]          # What the tool returned (from tool_node)
    rag_chunks: Optional[list]           # What RAG retrieved (from rag_node)
    
    # ── Final output ────────────────────────────────────────
    final_response: str                  # What we tell the customer (from respond_node)
```

### Why TypedDict and not a regular dict?

Compare:

```python
# Regular dict — no guardrails
state = {}
state["intet"] = "refund"     # typo → silent bug, discovered at 2am
state["urgency"] = 42         # wrong type → breaks downstream, discovered in production

# TypedDict — your IDE catches mistakes
class AgentState(TypedDict):
    intent: str
    urgency: str

state: AgentState = {"intet": "refund"}  # IDE screams at you immediately
```

**TypedDict gives you:**
1. **Autocomplete** — your IDE knows every field
2. **Type checking** — wrong types get caught before runtime
3. **Documentation** — anyone reading the code knows the full shape of state

### Why these specific fields?

Let's trace each one back to your manual loop:

| Field | Where it came from in manual loop | Why it's in state |
|---|---|---|
| `user_message` | `user_input` variable | Every node needs to know what the customer said |
| `intent` | `state["last_intent"]` (line 106) | Router reads it to decide the path |
| `urgency` | `state["last_urgency"]` (line 107) | Router checks for "critical" → escalate |
| `confidence` | `state["last_confidence"]` (line 109) | Router checks < 0.60 → escalate |
| `route` | `route` local variable (line 346) | For logging/debugging which path was taken |
| `tool_called` | `state["last_tool_called"]` (line 167) | respond_node uses this to craft the response |
| `tool_result` | `state["last_tool_result"]` (line 168) | respond_node uses this data |
| `final_response` | `agent_response` variable (line 333) | The output — what gets shown to the user |

> **Nothing is random. Every field exists because a node needs to read it or write it.**

---

## Part 3: Nodes — Building Each Kitchen Station

### The Golden Rule of Nodes

```
A node is a function that:
  1. Takes the FULL state
  2. Does ONE thing
  3. Returns ONLY the fields it changed (partial dict)
```

LangGraph handles merging the partial dict back into the full state.

---

### Node 1: classify_node — The Ticket Taker

This is your Skill 1 (`ticket_classifier.py`) wrapped as a LangGraph node.

```python
from ticket_classifier import create_classifier, classify_ticket

# Create the client ONCE, outside the node (shared resource)
classifier_client = create_classifier()

def classify_node(state: AgentState) -> dict:
    """
    READS:  state["user_message"]
    WRITES: intent, urgency, sentiment, confidence
    
    This is the FIRST node in the graph. It answers:
    "What does the customer want, and how urgent is it?"
    """
    result = classify_ticket(classifier_client, state["user_message"])
    
    return {
        "intent":     result.intent.value,
        "urgency":    result.urgency.value,
        "sentiment":  result.sentiment.value,
        "confidence": result.confidence,
    }
```

**Architectural choices to notice:**

1. **`classifier_client` is created outside the node.**
   Why? Because `classify_node` gets called on EVERY message. Creating a new client each
   time wastes time and memory. The client is stateless — one instance serves all calls.

2. **We return `.value` (strings), not the Enum objects.**
   Why? Because state is serializable. If you want to save state to a database (Skill 6),
   enums don't serialize cleanly. Strings do. Always store primitives in state.

3. **We return EXACTLY 4 fields.**
   Not the whole state. Not extra stuff "just in case." Exactly what this node is
   responsible for. If classify_node is writing `tool_result`, something is architecturally
   wrong.

---

### Node 2: tool_node — The Grill Station

This is your Skill 2 (`tool_calling_agent.py`) wrapped as a node.

```python
from tool_calling_agent import create_agent, run_agent

tool_client = create_agent()

def tool_node(state: AgentState) -> dict:
    """
    READS:  state["user_message"]
    WRITES: tool_called, tool_result, final_response
    
    Called when intent is order_status or refund_request.
    Delegates to Skill 2's run_agent() which handles the full
    tool-calling loop (LLM picks tool → execute → LLM responds).
    """
    result = run_agent(tool_client, state["user_message"])
    
    return {
        "tool_called":    result["tool_called"],
        "tool_result":    result["tool_result"],
        "final_response": result["final_response"],
    }
```

**Why does tool_node write `final_response` directly?**

Because Skill 2's `run_agent()` already handles the full loop:
1. LLM picks a tool
2. Tool executes
3. Result goes back to LLM
4. LLM writes the final customer-facing response

The tool node doesn't need a separate `respond_node` after it — the response is baked in.

> In a more advanced architecture, you'd separate "call the tool" from "write the response."
> For now, we keep it simple because your Skill 2 already handles both.

---

### Node 3: rag_node — The Sauce Station

This is your Skill 3 (`rag_pipeline.py`) wrapped as a node.

```python
# Only import if RAG is available (same pattern as manual loop)
try:
    from rag_pipeline import get_connection, setup_table, query as rag_query
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

def rag_node(state: AgentState) -> dict:
    """
    READS:  state["user_message"]
    WRITES: rag_chunks, final_response
    
    Called when intent is product_question or complaint (non-critical).
    Searches the knowledge base and generates a grounded answer.
    """
    if not RAG_AVAILABLE:
        return {
            "rag_chunks":     [],
            "final_response": "I don't have access to the knowledge base right now. "
                            "Let me connect you with a team member who can help.",
        }
    
    # rag_query handles: embed → retrieve → generate answer
    rag_result = rag_query(genai_client, conn, state["user_message"], verbose=False)
    
    return {
        "rag_chunks":     rag_result.sources,
        "final_response": rag_result.answer,
    }
```

**Why check `RAG_AVAILABLE` inside the node?**

Because the node might be called even if RAG isn't set up (e.g., in testing).
A node should NEVER crash. It should always return a valid partial state.
If it can't do its job, it returns a graceful fallback.

> **Production rule: A node that crashes kills the whole graph.**
> A node that returns a fallback keeps the agent alive.

---

### Node 4: escalate_node — The Emergency Station

```python
def escalate_node(state: AgentState) -> dict:
    """
    READS:  state["user_message"], state["urgency"]
    WRITES: tool_called, tool_result, final_response
    
    Called when urgency is critical OR confidence < 0.60.
    This is the safety net — when the agent can't handle it,
    hand off to a human.
    """
    from tool_calling_agent import escalate_to_human
    
    result = escalate_to_human(
        reason=f"Customer message: {state['user_message']}",
        urgency=state.get("urgency", "high"),
    )
    
    return {
        "tool_called":    "escalate_to_human",
        "tool_result":    result,
        "final_response": f"I'm connecting you with a human agent right away. "
                         f"Your case ID is {result['escalation_id']}. "
                         f"Estimated wait: {result['estimated_wait_time']}.",
    }
```

**Why call the tool function directly instead of going through Skill 2's `run_agent()`?**

In your manual loop (line 207-211), you had to HACK the message to force escalation:

```python
# manual_agent_loop.py — the hack
escalation_message = f"I need to escalate this conversation to a human agent immediately..."
result = run_agent(tool_agent_client, escalation_message)  # hope the LLM picks the right tool
```

That's fragile. The LLM might NOT pick `escalate_to_human`. It might pick `order_lookup`
instead. You're hoping the prompt hack works.

In LangGraph, the escalation node **calls the function directly**. No LLM in the loop.
No hoping. No hacks. Deterministic.

> **Architecture principle:** Use LLMs for decisions that NEED intelligence.
> Use direct function calls for actions that are ALREADY decided.

---

### Node 5: direct_node — The Simple Counter

```python
from google import genai
from google.genai import types

genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

DIRECT_RESPONSE_PROMPT = """You are a friendly customer support agent for Acme Corp.
The customer has sent a general message that doesn't require looking up orders,
processing refunds, or searching a knowledge base.
Respond naturally, warmly, and concisely. Keep it under 3 sentences."""

def direct_node(state: AgentState) -> dict:
    """
    READS:  state["user_message"]
    WRITES: final_response
    
    Called for greetings, feedback, general questions.
    Simple LLM response — no tools, no RAG.
    """
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=state["user_message"],
        config=types.GenerateContentConfig(
            system_instruction=DIRECT_RESPONSE_PROMPT,
            temperature=0.7,
        ),
    )
    
    return {"final_response": response.text}
```

**Why a separate node instead of handling it inline?**

Because it's **testable in isolation**. You can call `direct_node({"user_message": "hello"})`
and verify it responds correctly — without running the whole graph.

---

## Part 4: The Router — The Traffic Controller

The router is NOT a node. It's a **pure function** that reads state and returns a string.
That string tells LangGraph which node to go to next.

```python
def route_message(state: AgentState) -> str:
    """
    READS: state["urgency"], state["confidence"], state["intent"]
    RETURNS: string — the name of the next node
    
    This is the EXACT same logic as step_route() in your manual loop.
    But now it's a clean, isolated function.
    
    Decision tree:
      critical urgency  → "escalate"
      low confidence    → "escalate"
      order/refund      → "tool"
      product/complaint → "rag"
      everything else   → "direct"
    """
    # Safety first — escalate critical situations
    if state["urgency"] == "critical":
        return "escalate"
    
    # Low confidence — agent isn't sure, hand to human
    if state["confidence"] < 0.60:
        return "escalate"
    
    # Order-related — use tools
    if state["intent"] in ("order_status", "refund_request"):
        return "tool"
    
    # Knowledge-related — use RAG
    if state["intent"] in ("product_question", "complaint"):
        return "rag"
    
    # Everything else — simple response
    return "direct"
```

**Why isn't the router a node?**

Because it **doesn't change state**. It only READS state and returns a direction.
LangGraph keeps this distinction clean:
- **Nodes** = do work, change state
- **Router functions** = read state, pick a path (no side effects)

Compare to your manual loop:

```python
# manual_agent_loop.py — router was a function too (step_route)
# But it was tangled with classification objects:
def step_route(classification: TicketClassification, use_rag: bool) -> str:
    intent = classification.intent.value     # reaching into the object
    urgency = classification.urgency.value   # reaching into the object
```

In LangGraph, the router reads from **state** — the single source of truth.
It doesn't need the classification object. It doesn't need the `use_rag` flag.
Everything is already in state.

---

## Part 5: Wiring the Graph — Connecting the Stations

This is where the magic happens. 10 lines that replace 100+ lines of manual routing.

```python
from langgraph.graph import StateGraph, START, END

def build_graph():
    """
    Build the customer support agent graph.
    
    This function IS the architecture diagram. Read it top to bottom
    and you see the entire flow.
    """
    
    # ── Step 1: Create the graph with our state shape ────────
    graph = StateGraph(AgentState)
    
    # WHY: StateGraph needs to know the shape of state
    # so it can merge partial dicts from each node correctly.
    
    
    # ── Step 2: Add all nodes (kitchen stations) ─────────────
    graph.add_node("classify",  classify_node)
    graph.add_node("tool",      tool_node)
    graph.add_node("rag",       rag_node)
    graph.add_node("escalate",  escalate_node)
    graph.add_node("direct",    direct_node)
    
    # WHY: Each add_node registers a function.
    # The string name is how edges refer to it.
    # The function is what runs when the graph reaches this node.
    
    
    # ── Step 3: Set the entry point ──────────────────────────
    graph.add_edge(START, "classify")
    
    # WHY: Every message starts at classification.
    # START is a special LangGraph constant meaning "beginning of the graph."
    # This replaces: classification = step_classify(...) being always first.
    
    
    # ── Step 4: Add conditional routing after classify ───────
    graph.add_conditional_edges(
        "classify",          # FROM this node...
        route_message,       # ...CALL this function to decide...
        {                    # ...and go to whichever node it returns:
            "tool":     "tool",
            "rag":      "rag",
            "escalate": "escalate",
            "direct":   "direct",
        }
    )
    
    # WHY: This replaces the entire if/elif/elif tree in your manual loop.
    # The dict maps router return values → node names.
    # Adding a new route = adding ONE line here + writing the new node.
    
    
    # ── Step 5: All action nodes lead to END ─────────────────
    graph.add_edge("tool",     END)
    graph.add_edge("rag",      END)
    graph.add_edge("escalate", END)
    graph.add_edge("direct",   END)
    
    # WHY: After any action node runs, the graph is done.
    # END is a special LangGraph constant meaning "stop here."
    # Each action node already writes final_response, so we're done.
    
    
    # ── Step 6: Compile the graph ────────────────────────────
    app = graph.compile()
    
    # WHY: compile() locks the graph — no more adding nodes or edges.
    # It returns a runnable object that you call with app.invoke().
    # Think of it like compiling code: the graph is "source," the app is "binary."
    
    return app
```

### The graph, visualized:

```
                          ┌──────────┐
                    ┌────→│ tool     │────→ END
                    │     └──────────┘
┌───────┐    ┌─────┴────┐ ┌──────────┐
│ START  │───→│ classify │─→│ rag      │────→ END
└───────┘    └─────┬────┘ └──────────┘
                    │     ┌──────────┐
                    ├────→│ escalate │────→ END
                    │     └──────────┘
                    │     ┌──────────┐
                    └────→│ direct   │────→ END
                          └──────────┘
```

**Compare this to your manual loop.** Can you see the flow in `manual_agent_loop.py`
without reading every line? No. Can you see it here? Yes. That's the point.

---

## Part 6: Running It — Placing the Order

```python
def run(user_message: str) -> str:
    """
    Run the agent for a single user message.
    
    This is the equivalent of ONE TURN in your manual loop.
    """
    app = build_graph()
    
    # Create the initial state — the order ticket
    initial_state: AgentState = {
        "user_message":   user_message,
        "intent":         "",
        "urgency":        "",
        "sentiment":      "",
        "confidence":     0.0,
        "route":          "",
        "tool_called":    None,
        "tool_result":    None,
        "rag_chunks":     None,
        "final_response": "",
    }
    
    # Place the order — the kitchen handles the rest
    final_state = app.invoke(initial_state)
    
    return final_state["final_response"]
```

**What `app.invoke()` does under the hood:**

```
1. Starts at START
2. Runs classify_node(initial_state)
   → state now has: intent, urgency, sentiment, confidence
3. Calls route_message(state)
   → returns "tool" (for example)
4. Runs tool_node(state)
   → state now has: tool_called, tool_result, final_response
5. Reaches END
6. Returns the final state
```

That's it. The entire manual loop — classify, route, act, respond — handled by
`app.invoke()` in one line.

---

## Part 7: The Full File — Everything Together

When you're ready to build, we'll create `langgraph_agent.py` with all of this
wired together. The structure will be:

```
langgraph_agent.py
├── Imports
├── AgentState (TypedDict)
├── Node functions:
│   ├── classify_node()
│   ├── tool_node()
│   ├── rag_node()
│   ├── escalate_node()
│   └── direct_node()
├── Router function:
│   └── route_message()
├── Graph builder:
│   └── build_graph()
├── Runner:
│   └── run()
└── Main:
    └── Interactive loop + test cases
```

---

## Part 8: What You Should Understand Before We Code

Before we write the actual `langgraph_agent.py`, make sure these click:

### Checklist — Can you answer these?

1. **Why is AgentState a TypedDict and not a regular dict?**
   _(Type safety, autocomplete, documentation)_

2. **Why do nodes return partial dicts instead of the whole state?**
   _(Separation of concerns — each node only writes what it changed)_

3. **Why is the router a separate function and not a node?**
   _(It doesn't change state — it only reads and decides)_

4. **Why does escalate_node call the function directly instead of through the LLM?**
   _(The decision is already made — no need for LLM intelligence)_

5. **Why do we create clients (classifier_client, tool_client) outside the nodes?**
   _(Performance — avoid recreating on every message)_

6. **What happens if a node crashes?**
   _(The whole graph fails — that's why nodes must return graceful fallbacks)_

---

## What's Next

When you're ready, say **"Let's build it"** and we'll create the actual
`langgraph_agent.py` file — production-quality code with every line explained.

We'll also run it against 30 test scenarios to hit the 85%+ target from your
implementation plan.

> You've seen the blueprint. Now we build the house. 🏗️
