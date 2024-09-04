
def filter_graph(instances, graph, thresh=0.5):
    if instances['scores'].shape[0] <= 1:
        return graph
    keep = instances['scores'] > thresh
    rel, ix = graph
    keep = keep[ix[0]] & keep[ix[1]]
    keep = keep.cpu().numpy()
    rel = rel[keep]
    ix = ix[:, keep]
    return rel, ix

def filter_boxes(instances, thresh=0.5):
    keep = instances['scores'] > thresh
    instances = {k: v[keep] for k, v in instances.items()}
    return instances