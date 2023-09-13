# MARL Testbed in Multi-floor Transportation Problem

Welcome to the MARL (Multi-Agent Reinforcement Learning) Testbed tailored for multi-floor transportation challenges. This testbed simulates the environment of a multi-floor smart factory, where multiple transportation robots are tasked with the job of pick and delivery.

## Problem Description

In our model of the smart factory, we have implemented the following constraints:

- **Path Capacity**: Each path connecting two locations has a capacity of 1. This implies that at any given time, only one robot can traverse a path in a specific direction.

This setting introduces complexities as robots must learn to coordinate and negotiate their routes to prevent conflicts and ensure smooth operation.

## Key Features

1. **Algorithms**: We have implemented two widely recognized MARL algorithms - QMIX and MADDPG. 
2. **Visualizations**: A plotting file is provided to visualize and evaluate the performance of the implemented algorithms in this environment.

## Installation and Dependencies

- The project relies on `ray` version `2.2.0`. Please note that the support for `Gymnasium` has not been added.

## Results

Utilize the provided plot file to observe the results of the agents' performance over time. This visualization will help understand how different MARL algorithms respond to the unique challenges of our multi-floor transportation problem.