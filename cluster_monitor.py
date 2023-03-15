def monitor_cluster_health(nodes):
    """Monitoring logic for large scale AI training clusters."""
    for node in nodes:
        status = check_node_telemetry(node)
        if status != 'HEALTHY':
            isolate_node(node)
    return True
