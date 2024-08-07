import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np
import copy
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description="Simulate rabbit and fox population")
parser.add_argument('-g', '--grass-growth-rate', type=float, default=1.0, help="Grass growth rate")
parser.add_argument('-k', '--fox-k-value', type=float, default=1.0, help="Fox k value")
parser.add_argument('-s', '--field-size', type=int, default=20, help="Field size")
parser.add_argument('-r', '--initial-rabbits', type=int, default=10, help="Initial number of rabbits")
parser.add_argument('-f', '--initial-foxes', type=int, default=5, help="Initial number of foxes")
args = parser.parse_args()

SIZE = args.field_size  # The dimensions of the field
INITIAL_FOXES = args.initial_foxes
INITIAL_RABBITS = args.initial_rabbits
FOXES_K = args.fox_k_value
GRASS_RATE = args.grass_growth_rate  # Probability that grass grows back at any location in the next season.
OFFSPRING = 3  # Max offspring offspring when a rabbit reproduces
WRAP = False  # Does the field wrap around on itself when rabbits move?
"""MAX_HUNGER = 10"""


class Rabbit:
    """ A furry creature roaming a field in search of grass to eat.
    Mr. Rabbit must eat enough to reproduce, otherwise he will starve. """

    def __init__(self):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0

    def reproduce(self):
        """ Make a new rabbit at the same location.
         Reproduction is hard work! Each reproducing
         rabbit's eaten level is reset to zero. """
        self.eaten = 0
        return copy.deepcopy(self)

    def eat(self, amount):
        """ Feed the rabbit some grass """
        self.eaten += amount

    def move(self):
        """ Move up, down, left, right randomly """

        if WRAP:
            self.x = (self.x + rnd.choice([-1, 0, 1])) % SIZE
            self.y = (self.y + rnd.choice([-1, 0, 1])) % SIZE
        else:
            self.x = min(SIZE - 1, max(0, (self.x + rnd.choice([-1, 0, 1]))))
            self.y = min(SIZE - 1, max(0, (self.y + rnd.choice([-1, 0, 1]))))

class Fox:
    """ A furry creature roaming a field in search of rabbit to eat.
    Mr. FOX must eat enough to reproduce, otherwise he will starve. """

    def __init__(self):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0
        self.k = FOXES_K
        """self.hunger = MAX_HUNGER"""

    def eat(self, rabbits):
        """ Feed the fox """
        eaten_rabbits = []
        for rabbit in rabbits:
            if self.x == rabbit.x and self.y == rabbit.y:
                self.eaten += 1
                eaten_rabbits.append(rabbit)
                self.k = 10
        return eaten_rabbits

    def survive(self):
        """ Check if the fox survives """
        self.k -= 1
        return self.k >= 0

    def reproduce(self):
        """ Make a new fox at the same location.
         Reproduction is hard work! Each reproducing
         fox's eaten level is reset to zero. """
        if self.eaten >= 1:
            self.eaten = 0
            return copy.deepcopy(self)
        else:
            return None

    def move(self):
        """ Move up, down, left, right randomly """

        if WRAP:
            self.x = (self.x + rnd.choice([-1, 0, 1])) % SIZE
            self.y = (self.y + rnd.choice([-1, 0, 1])) % SIZE
        else:
            self.x = min(SIZE - 1, max(0, (self.x + rnd.choice([-2, 0, 2]))))
            self.y = min(SIZE - 1, max(0, (self.y + rnd.choice([-2, 0, 2]))))


class Field:
    """ A field is a patch of grass with 0 or more rabbits hopping around
    in search of grass """

    def __init__(self):
        """ Create a patch of grass with dimensions SIZE x SIZE
        and initially no rabbits """
        self.rabbits = []
        self.field = np.ones(shape=(SIZE, SIZE), dtype=int)
        self.nrabbits = []
        self.ngrass = []
        self.foxes = []
        self.nfoxes = []

    def add_rabbit(self, rabbit):
        """ A new rabbit is added to the field """
        self.rabbits.append(rabbit)

    def add_fox(self, fox):
        """ A new rabbit is added to the field """
        self.foxes.append(fox)

    def move(self):
        """ Rabbits move """
        for r in self.rabbits:
            r.move()
        for f in self.foxes:
            f.move()

    def eat(self):
        """ Rabbits eat (if they find grass where they are) """

        for rabbit in self.rabbits:
            rabbit.eat(self.field[rabbit.x, rabbit.y])
            self.field[rabbit.x, rabbit.y] = 0
        self.eat_rabbits()
        """for fox in self.foxes:
            eaten_rabbits = fox.eat(self.rabbits)
            for rabbit in eaten_rabbits:
                self.rabbits.remove(rabbit)"""

    def eat_rabbits(self):
        """Foxes must eat rabbits to survive"""
        for fox in self.foxes:
            for rabbit in self.rabbits:
                if fox.x == rabbit.x and fox.y == rabbit.y:
                    fox.eat(self.rabbits)
                    fox.k = FOXES_K
                    self.rabbits.remove(rabbit)
                    break

    def survive(self):
        """ Rabbits who eat some grass live to eat another day """
        self.rabbits = [r for r in self.rabbits if r.eaten > 0]
        self.foxes = [f for f in self.foxes if f.survive() and f.k > 0]

    """def starve_foxes(self):
        self.foxes = [f for f in self.foxes if f.hunger > 0]"""

    def reproduce(self):
        """ Rabbits reproduce like rabbits. """
        born = []
        for rabbit in self.rabbits:
            for _ in range(rnd.randint(1, OFFSPRING)):
                born.append(rabbit.reproduce())
        self.rabbits += born

        born_foxes = []
        for fox in self.foxes:
            new_fox = fox.reproduce()
            if new_fox:  # Check if the fox has reproduced
                born_foxes.append(new_fox)
        self.foxes += born_foxes

        # Capture field state for historical tracking
        self.nrabbits.append(self.num_rabbits())
        self.ngrass.append(self.amount_of_grass() / 10)
        self.nfoxes.append(self.num_foxes())

    def grow(self):
        """ Grass grows back with some probability """
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def get_rabbits(self):
        rabbits = np.zeros(shape=(SIZE, SIZE), dtype=int)
        for r in self.rabbits:
            rabbits[r.x, r.y] = 1
        return rabbits

    def num_rabbits(self):
        """ How many rabbits are there in the field ? """
        return len(self.rabbits)

    def amount_of_grass(self):
        return self.field.sum()

    def num_foxes(self):
        return len(self.foxes)

    def get_field_state(self):
        state = np.zeros(shape=(SIZE, SIZE), dtype=int)
        for r in self.rabbits:
            state[r.x, r.y] = 2
        for f in self.foxes:
            state[f.x, f.y] = 3
        state += self.field
        return state

    def generation(self):
        """ Run one generation of rabbits """
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        # self.eat_rabbits()
        # self.starve_foxes()
        self.grow()

    def history(self, showTrack=True, showPercentage=True, marker='.'):

        plt.figure(figsize=(6, 6))
        plt.xlabel("# Rabbits")
        plt.ylabel("# Grass")

        xs = self.nrabbits[:]
        if showPercentage:
            maxrabbit = max(xs)
            xs = [x / maxrabbit for x in xs]
            plt.xlabel("% Rabbits")

        ys = self.ngrass[:]
        if showPercentage:
            maxgrass = max(ys)
            ys = [y / maxgrass for y in ys]
            plt.ylabel("% Rabbits")

        if showTrack:
            plt.plot(xs, ys, marker=marker)
        else:
            plt.scatter(xs, ys, marker=marker)

        plt.grid()

        plt.title("Rabbits vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        plt.savefig("history.png", bbox_inches='tight')
        plt.show()

    def history2(self):
        xs = self.nrabbits[:]
        ys = self.ngrass[:]

        sns.set_style('dark')
        f, ax = plt.subplots(figsize=(7, 6))

        sns.scatterplot(x=xs, y=ys, s=5, color=".15")
        sns.histplot(x=xs, y=ys, bins=50, pthresh=.1, cmap="mako")
        sns.kdeplot(x=xs, y=ys, levels=5, color="r", linewidths=1)
        plt.grid()
        plt.xlim(0, max(xs) * 1.2)

        plt.xlabel("# Rabbits")
        plt.ylabel("# Grass")
        plt.title("Rabbits vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        plt.savefig("history2.png", bbox_inches='tight')
        plt.show()

    def popgens(self):
        grass_count = list_as_percents(self.ngrass[:])
        fox_count = list_as_percents(self.nfoxes[:])
        rabbit_count = list_as_percents(self.nrabbits[:])
        print(self.nfoxes)

        generations = list(range(len(grass_count)))

        # Create a figure and axis object
        fig, ax = plt.subplots()

        # Plot each line with a label and color
        ax.plot(generations, grass_count, label='Grass', color='green')
        ax.plot(generations, rabbit_count, label='Rabbits', color='blue')
        ax.plot(generations, fox_count, label='Foxes', color='red')

        # Set the title and axis labels
        ax.set_title('Populations After 1000+ Generations')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Population (% max)')

        # Add a legend
        ax.legend()

        # Display the plot
        plt.show()


def list_as_percents(l):
    """Helper for graphing List[num] --> List[% of max]"""
    peak = max(l)
    for i in range(len(l)):
        l[i] = (l[i]/peak)
    return l


def animate(i, field, im):
    field.generation()
    # print("AFTER: ", i, np.sum(field.field), len(field.rabbits))
    field_state = field.get_field_state()
    im.set_array(field_state)
    plt.title("generation = " + str(i))
    return im,


def main():
    # Create the ecosystem
    field = Field()

    for _ in range(INITIAL_RABBITS):
        field.add_rabbit(Rabbit())

    # Add initial foxes to the field
    for _ in range(INITIAL_FOXES):
        field.add_fox(Fox())

    cmap = mcolors.ListedColormap(['tan', 'red', 'blue', 'green'])

    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(array, cmap=cmap, interpolation='hamming', aspect='auto', vmin=0, vmax=1)
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im,), frames=1000000, interval=1, repeat=True)
    plt.show()

    """field.history()
    field.history2()"""
    field.popgens()


if __name__ == '__main__':
    main()

"""
def main():

    # Create the ecosystem
    field = Field()
    for _ in range(10):
        field.add_rabbit(Rabbit())


    # Run the world
    gen = 0

    while gen < 500:
        field.display(gen)
        if gen % 100 == 0:
            print(gen, field.num_rabbits(), field.amount_of_grass())
        field.move()
        field.eat()
        field.survive()
        field.reproduce()
        field.grow()
        gen += 1

    plt.show()
    field.plot()

if __name__ == '__main__':
    main()


"""





