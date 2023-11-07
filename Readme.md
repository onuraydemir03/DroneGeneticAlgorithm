# Genetic Algorithm that Explores a Map

This algorithm runs N number of agents synchronously and discover a map. The rules of algorithm are behing.

### Rules
- Agents should not pass same position
- Agents should not rotate too much, it is a waste of energy to redirect
- Agents should not go out of map

![](Animation.gif)


- NUM_OF_MOVES: How many step the agents will move
- NUM_OF_CHROMOSOMES_IN_ONE_SET: How many agent will work together synchronously
- POPULATION_SIZE: Size of the population
- MUTATION_RATE: Random gene changing rate
- DROP_BAD_RATE: How many chromosome set will be dropped from the population at one iter


# Hyperparameters
~~~
Edit config.py file
~~~

# Running
~~~
python3 discover.py
~~~