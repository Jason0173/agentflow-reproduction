from agentflow.agentflow.solver import construct_solver

llm_engine_name = "together-Qwen/Qwen2.5-7B-Instruct-Turbo"

solver = construct_solver(
    llm_engine_name=llm_engine_name,
    model_engine=["trainable", "trainable", "trainable", "trainable"],
)

output = solver.solve("What is the capital of France?")
print(output["direct_output"])
