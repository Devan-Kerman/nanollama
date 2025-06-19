# All To One

## Tree-Based Buffer Reduction

Each device has a tensor buffer and a reduction buffer. When performing a reduction, we reduce in a tree, where each
children sends their tensor to the reduction buffer

## Pathfinding Algorithm

First, choose your destination node. Then partition your grid into it's parts and choose the closest node in each
partition to your node. Then for each of those nodes, partition again and so on, so forth.

TODO make markdown table graphic of this