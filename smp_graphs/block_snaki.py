import tensorflow as tf

from Agent import Agent, Memory, PMemory
from Environment import SnakeEnvironment as SE
from Gui import SnakeGui

from Layer import InputLayer, CNNLayer, MLPLayer, OutputLayer
from QNet import DQN, DDQN, DuelQN

import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from smp_graphs.block import decStep, decInit, Block2, PrimBlock2

class SnakiBlock2(PrimBlock2):
    @decInit()
    def __init__(self, conf, paren, top):
        PrimBlock2.__init__(self, conf=conf, paren=paren, top = top)

        self.N = 200000
        self.hyper = {
            "epoch": self.N,
            "assign": 10000,
            "batch": lambda x: int(((1 - 56) / self.N) * x + 56) + 8,
            "maze": (7, 7),
            "frames": 1,
            "grows": 1,
            "walls": True,
            "Rdie": -1,
            "Rlife": -0.0,
            "Rgood": 1,
            "Memory": 2 ** 20,
            "sort_freq": 1000,
            "q": 0.9999,
            "Convs": [[1, 1, 1, 1]],
            "MLP": [4, 4, 80, 80, 4, 4],
            "Output": [4, 4],
            "eps": lambda x:
            0.6 * (1 - x / self.N) + 0.15,
        # * 1 / 2 * (1 + np.cos((2 * np.pi * x * 5) /
            # self.N)) + 0.15,
            "gamma": 0.7,
            "activation": tf.nn.relu,
            "Optimizer": tf.train.RMSPropOptimizer(0.01, momentum=0.5)
        }

        logger.debug("Hyperparameters: %s" % self.hyper)
        # Environment -------------------------------------------------------------
        logger.info("Load Environment.")
        self.maze = self.hyper["maze"]

        self.env = SE(
            dims=(self.maze[0], self.maze[1]),
            grows=self.hyper["grows"],
            walls=self.hyper["walls"],
            frames=self.hyper["frames"],
            appear=SE.STATE_COORD,
            reward={"die": self.hyper["Rdie"],
                    "life": self.hyper["Rlife"],
                    "good": self.hyper["Rgood"]}
        )

        # Memory ------------------------------------------------------------------
        logger.info("Load Memory.")
        self.mem = Memory(
            size=self.hyper["Memory"],
            # state_shape=(self.hyper["frames"], self.maze[0], self.maze[1]),
            state_shape=(self.hyper["frames"], 2, self.hyper["grows"] + 1)
            # ,sort_freq=self.hyper["sort_freq"]
        )

        self.mem.load()
        logger.debug("Memory size: %d" % len(self.mem))

        # Brain -------------------------------------------------------------------
        logger.info("Load Q-Network")
        self.inputL = InputLayer(
            dims=[self.hyper["frames"], 2, self.hyper["grows"] + 1, 1]
        )
        self.convL = CNNLayer(
            dims=self.hyper["Convs"],
            activation=self.hyper["activation"],
            stddev=1
        )
        self.mlpL = MLPLayer(
            dims=self.hyper["MLP"],
            activation=self.hyper["activation"],
            stddev=1
        )
        self.outputL = OutputLayer(
            dims=self.hyper["Output"],
            stddev=1
        )

        self.Q = DDQN(
            layers=[self.inputL, self.convL, self.mlpL, self.outputL],
            optimizer=self.hyper["Optimizer"]
        )

        # Agent -------------------------------------------------------------------
        logger.info("Create Agent")
        self.snaKI = Agent(
            env=self.env,
            mem=self.mem,
            net=self.Q,
            eps=self.hyper["eps"],
            gamma=self.hyper["gamma"]
        )

        logger.info("Start GUI")
        self.gui = SnakeGui(None)

        # Please forgive a little bug. Separate creation from learning:------------

        # TODO make both simultaneous possible

        logger.info("Do the business")
        self.create(self.gui)
        # learn(snaKI)

    @decStep()
    def step(self, x = None):
        print "self.cnt", self.cnt
        self.learn(None)
        # "eps": lambda x: 0.7*(1 - (x / float(self.N)) + np.sin(x / (self.N / (17. * 2 * np.pi))) / 10.)


    def create(self, gui):
        """
        Create Memory by human agent via api.
        :param gui: Gui to play in
        :return:
        """
        logger.info("Starting GUI to create memory")

        self.gui.start()

        logger.debug("Memory:\n %s" % self.mem)
        logger.debug("Memory size: %d" % len(self.mem))
        logger.info("Save Memory")

        self.mem.save()

    def learn(self, snaki):
        """Let Snaki learn"""
        logger.info("Start learning process.")

        self.snaki.learn(train_len=self.hyper["epoch"],
                    assign_freq=self.hyper["assign"],
                    batch=self.hyper["batch"],
                    restore=not True,
                    q=self.hyper["q"])

        # take a look how snaki is doing
        logger.info("Start GUI (autonomously)...")
        self.gui.autonom = True
        self.gui.start()
