### <font color=black> <font size = 5>Topics to Cover:</font>
<font color=blue> <font size = 3>
> 1. **What is deap**
> 2. **Creator Module**
> 3. **Toolbox Module**
> 4. **Default Algos/Functions**:
</font>



```python
### importing all the libraries 
import numpy as np
from deap import creator, base, tools, algorithms
```

***

### <font color=green> <font size=6>1. What is Deap?</font>

<font color=red> <font size = 3>**DEAP is a evolutionary computation framework which can be used to design/prototype your own custom GAs** 

**For more documentation, visit: https://deap.readthedocs.io/en/master/**

**Github Link: https://github.com/DEAP/deap**
    </font>

### <font color=green> <font size=6>2. Creator Module</font>

<font color=red> <font size = 3> **We will use this module to define our own custom classes (data types)**</font></font>
  
  
<font color=red> <font size = 3> **For all the GAs we wil be building two classes, one for the fitness and other for the individual**</font></font>

<font color=red> <font size = 3> **This takes in the <font color=blue>*name of the class/data type you want to define*</font>, <font color=blue>*parent class from which you want to inherit*</font> and  <font color=blue>*init arguments*</font>**</font></font>

<font color=red> <font size = 3> **Please find the sample snippet below:**</font></font>




```python
## We will define a class with name Indivdual which inherits from list and has an attribute fitness
creator.create("sample_class", list, fitness=100)
```


```python
### let's use this data type, we will declare one indidual using this
sample=creator.sample_class([0,0,0,1,0])
```


```python
### we can see ind has an attribute of fitness and also inhertis all the methods of list class
sample.fitness,sample,sample.append(1),sample
```




    (100, [0, 0, 0, 1, 0, 1], None, [0, 0, 0, 1, 0, 1])



<font color=green> <font size = 4> **We saw how to use creator to define custom datatypes, now we will see the two data types we will need to create our own custom GAs i.e. <font color=red>fitness and individual (which is very similar to what we defined above)</font>**</font></font>

<font color=green> <font size = 4> **we will use the code in the cell below in every GA we build**


```python
### defining a fitness class which inherits from a default deap class
### this takes a default argument of weights which is a tupple
### different scenarios of the weiht parameter:
### (1,)  MAXIMIZATION PROBLEM WE WANT TO MAXIMIZE THE OBJECTIVE FUNCTION
### (-1,) MINIMIZATION PROBLEM WE WANT TO MINIMIZE THE OBJECTIVE FUNCTION
### (1,-1) MULTIOBJECTIVE WEHRE WE WANT TO MAXIMIZE THE FIRST VALUE AND MINIMIZE THE SECOND OBJECTIVE

creator.create("fitness", base.Fitness, weights=(1.0,))
### individual inherting from list and having an attribute fitness which of the the type defined above
creator.create("Individual", list, fitness=creator.fitness)
```


```python
### example:
ind=creator.Individual([0,0,0,1])
ind.fitness ### this is blank and is of the class fitness defined above
```




    deap.creator.fitness(())



### <font color=green> <font size=6>3. Toolbox Module</font>

<font color=red> <font size = 3> **Toolbox gives us the flexibility to store all the functions in one place,as the name suggests it is a toolbox with all our tools init)**</font></font>
  
  
<font color=red> <font size = 3> **We will setup a toolbox and register different functions in it and call them using te toolbox**</font></font>




```python
### defining the toolbox
toolbox=base.Toolbox()

### defining the functions we want to register
def get_max_num(a):
    return np.max(a)
def get_min_num(a):
    return np.min(a)
def get_sum(a):
    return np.sum(a)
def get_sum_scale(a,scale):
    return np.sum(a)*scale


### registering the functions in the toolbox:
toolbox.register(alias='max_num',function=get_max_num)
toolbox.register(alias='min_num',function=get_min_num)
toolbox.register(alias='total',function=get_sum)
### example of a function taking in default argument values
toolbox.register(alias='scaled_total',function=get_sum_scale,scale=1.5) 
```


```python

```


```python
### defining a list to perform all these opertions on
rand_nums=np.random.randint(1,100,10).tolist()
rand_nums
```




    [55, 44, 76, 54, 16, 81, 93, 68, 57, 59]




```python
toolbox.max_num(rand_nums),toolbox.min_num(rand_nums),toolbox.total(rand_nums),toolbox.scaled_total(rand_nums)
```




    (93, 16, 603, 904.5)



<font color=red> <font size = 4> **We saw how we can setup a toolbox with different functions, in our GAs different functions we need to register are:**</font></font>
- <font color=green> <font size = 3> **Function creating the individuals (it maybe a list,dict or any other data type)** </font></font>
- <font color=green> <font size = 3> **Function for Crossover** </font></font>
- <font color=green> <font size = 3> **Function for Mutation** </font></font>
- <font color=green> <font size = 3> **Function for Selection** </font></font>    
- <font color=green> <font size = 3> **Function for Evaluation** </font></font>      
  
  


### <font color=green> <font size=6>4. Default Algos and Function</font>

<font color=red> <font size = 3> **In this section we will cover some default functions we will be using to build our GAs**</font></font>
  
  
<font color=red> <font size = 4> **Functions we will cover:**</font></font>
- <font color=green> <font size = 3> **one point crossover (CxOnePoint)** </font></font>
- <font color=green> <font size = 3> **Flip Bit Mutation (MutFlipBit)** </font></font>
- <font color=green> <font size = 3> **EA SIMPLE (default evolutionary algorithm by deap)** </font></font>


```python
### crossover example
toolbox.register('crossover',tools.cxOnePoint)

### defining the two individuals:
ind1=[0,1,0,1,0]
ind2=[0,0,1,1,0]

### results of crossover
toolbox.crossover(ind1,ind2)
```




    ([0, 0, 1, 1, 0], [0, 1, 0, 1, 0])




```python
### mutation example

toolbox.register('mutation',tools.mutFlipBit)

### defining the two individuals:
ind1=[0,1,0,1]

### results of crossover
###  indpd is a default parameter of mutflipbit, controls the prob at an element/gene level for mutation
print ('result when indpb=0.5 ',toolbox.mutation(ind1,indpb=0.5) ) 

ind1=[0,1,0,1]
print ('result when indpb=0 ',toolbox.mutation(ind1,indpb=0) ,' we see no gene/element was mutated because inpb was 0')

ind1=[0,1,0,1]
print ('result when indpb=1 ',toolbox.mutation(ind1,indpb=1),' we see every gene/element was mutated because inpb was 1')  

```

    result when indpb=0.5  ([1, 1, 0, 1],)
    result when indpb=0  ([0, 1, 0, 1],)  we see no gene/element was mutated because inpb was 0
    result when indpb=1  ([1, 0, 1, 0],)  we see every gene/element was mutated because inpb was 1


<font color=red> <font size = 3> **Pseudo code of the EA simple:**</font></font>

evaluate(population)
 
for g
in range(ngen):

    population = select(population, len(population))

    offspring = CrossOVer(population, cxpb) and Mutation (population, mutpb)

    evaluate(offspring)

    population = offspring
