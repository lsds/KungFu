#!/usr/bin/env python3

import yaml


def gen_network(net, subnet):
    return {
        net: {
            "ipam": {
                "driver": "default",
                "config": [
                    {
                        "subnet": subnet
                    },
                ],
            },
        },
    }


def gen_service(net, ip, tag, command):
    return {
        "image": tag,
        "command": command,
        "networks": {
            net: {
                "ipv4_address": ip,
            },
        },
    }


def gen_services(net, ips, tag, command):
    services = {}
    for i, ip in enumerate(ips):
        name = "node" + str(i)
        services[name] = gen_service(net, ip, tag, command)
    return services


def gen_compose(np, n_nodes, node_cap, tag, user_command):
    net = 'app'
    subnet = '172.16.238.0/24'

    ips = ["172.16.238.%d" % (10 + rank) for rank in range(n_nodes)]
    H = ','.join(ip + ":" + str(node_cap) for ip in ips)
    command = [
        "kungfu-run",
        "-timeout=1m",
        "-H",
        H,
        # "-self=127.0.0.1",
        "-nic=eth0",
        "-np",
        str(np),
        "-w",
    ] + user_command
    compose = {
        "version": "3",
        "services": gen_services(net, ips, tag, command),
        "networks": gen_network(net, subnet),
    }
    config = yaml.dump(compose)
    with open('docker-compose.yaml', 'w') as f:
        f.write(config)


def main():
    np = 16
    n_nodes = 4
    node_cap = 4
    tag = 'kungfu-adaptation-benchmark:snapshot'
    user_command = [
        'python3',
        'benchmarks/adaptive/adaptive_trainer.py',
    ]
    gen_compose(np, n_nodes, node_cap, tag, user_command)


main()
