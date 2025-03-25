import click
import grpc
from google.protobuf.empty_pb2 import Empty

from oobleck.elastic.master_service_pb2 import AgentInfo, DistInfo
from oobleck.elastic.master_service_pb2_grpc import OobleckMasterStub


@click.group(help="CLI tool to query and manipulate Oobleck training.")
@click.option("--ip", type=str, default="localhost", help="Master IP address.")
@click.option("--port", type=int, help="Master port.")
@click.pass_context
def main(ctx: click.core.Context, ip: str, port: int):
    channel = grpc.insecure_channel(f"{ip}:{port}")
    stub = OobleckMasterStub(channel)

    ctx.ensure_object(dict)
    ctx.obj["stub"] = stub


@main.command()
@click.pass_context
def get_agent_list(ctx: click.core.Context):
    stub: OobleckMasterStub = ctx.obj["stub"]
    response: DistInfo = stub.GetDistInfo(Empty())

    print("=== Agents ===")
    for index, host in enumerate(response.hosts):
        print(
            f"[{index}] IP: {host.ip}:{host.port} Status: {host.status} (device indices: {host.devices})"
        )
    print("==============")


@main.command()
@click.pass_context
@click.option("--agent_index", type=int)
def kill_agent(ctx: click.core.Context, agent_index: int):
    stub: OobleckMasterStub = ctx.obj["stub"]
    stub.KillAgent(AgentInfo(agent_index=agent_index))


if __name__ == "__main__":
    main(obj={})
