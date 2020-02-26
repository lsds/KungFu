#!/usr/bin/env python3

import argparse
import sys

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


IP_PREFIX = "172.16.238"


class IPPool:
    def __init__(self, prefix):
        self._prefix = prefix
        self._offset = 10

    def get(self):
        if self._offset >= 200:
            raise RuntimeError('insufficient IP')
        ip = "%s.%d" % (self._prefix, self._offset)
        self._offset += 1
        return ip


def gen_compose(np, n_nodes, node_cap, tag, user_command):
    net = 'app'
    subnet = '%s.0/24' % (IP_PREFIX)
    ip_pool = IPPool(IP_PREFIX)

    config_server_ip = ip_pool.get()
    config_client_ip = ip_pool.get()
    node_ips = [ip_pool.get() for _ in range(n_nodes)]
    H = ','.join(ip + ":" + str(node_cap) for ip in node_ips)

    command = [
        "kungfu-run",
        "-timeout=1m",
        "-q",
        "-H",
        H,
        "-nic=eth0",
        "-np",
        str(np),
        "-w",
    ] + user_command

    nodes = gen_services(net, node_ips, tag, command)

    config_server = gen_service(net, config_server_ip, tag, [
        'kungfu-peerlist-server',
        '-ttl',
        '10s',
    ])
    config_server['ports'] = ['9100:9100']

    config_client = gen_service(net, config_client_ip, tag, [
        'kungfu-peerlist-client',
        '-server',
        'http://%s:%d/put' % (config_server_ip, 9100),
        '-ttl',
        '10s',
        '-H',
        H,
    ])

    services = {
        'config_server': config_server,
        'config_client': config_client,
    }
    services.update(nodes)

    compose = {
        "version": "3",
        "services": services,
        "networks": gen_network(net, subnet),
    }
    config = yaml.dump(compose)
    with open('docker-compose.yaml', 'w') as f:
        f.write(config)


def parse_args():
    p = argparse.ArgumentParser(description='generate docker-compose.yaml')
    p.add_argument('--nodes', type=int, default=4, help='node count')
    p.add_argument('--node-cap', type=int, default=4, help='node cap')
    p.add_argument('--np', type=int, default=16, help='cluster size')
    p.add_argument('--image',
                   type=str,
                   default='kungfu-ci-base:snapshot',
                   help='docker image tag')
    return p.parse_args()


def main(args):
    user_command = [
        'python3',
        'benchmarks/adaptation/adaptive_trainer.py',
    ]
    gen_compose(args.np, args.nodes, args.node_cap, args.image, user_command)


main(parse_args())
