import asyncio
import inspect
import warnings
from typing import TypedDict, Optional

try:
    from agentops.sdk.decorators import operation
except ImportError:
    try:
        from agentops import record_action as operation
    except ImportError:
        def operation(func):
            return func

class RewardSpanData(TypedDict):
    type: "reward"
    value: Optional[float]

def reward(fn: callable) -> callable:
    def wrap_result(result: Optional[float]) -> RewardSpanData:
        if result is None:
            return {"type": "reward", "value": None}
        if not isinstance(result, (float, int)):
            warnings.warn(f"Reward is ignored: {result}")
            return {"type": "reward", "value": None}
        return {"type": "reward", "value": float(result)}

    is_async = asyncio.iscoroutinefunction(fn) or inspect.iscoroutinefunction(fn)

    if is_async:
        async def wrapper_async(*args, **kwargs):
            result: Optional[float] = None
            @operation
            async def agentops_reward_operation() -> RewardSpanData:
                nonlocal result
                result = await fn(*args, **kwargs)
                return wrap_result(result)
            await agentops_reward_operation()
            return result
        return wrapper_async
    else:
        def wrapper(*args, **kwargs):
            result: Optional[float] = None
            @operation
            def agentops_reward_operation() -> RewardSpanData:
                nonlocal result
                result = fn(*args, **kwargs)
                return wrap_result(result)
            agentops_reward_operation()
            return result
        return wrapper