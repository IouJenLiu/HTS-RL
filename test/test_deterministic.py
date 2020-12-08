import unittest
import subprocess
import sys, os
import pickle
import numpy as np


class TestDeterministic(unittest.TestCase):

    def setUp(self):

        ENVBIN = sys.exec_prefix
        BIN = os.path.join(ENVBIN, "bin", "python")
        print(BIN)
        subprocess.call([BIN, '../main.py', '--dump_run_id', '0', '--num-env-steps', '20500', '--dump_traj_flag',
                              '--eval-freq', '0', '--no-cuda', '--use-gae', '--num-processes', '8', '--num-actors', '8',
                              '--env-name', 'academy_3_vs_1_with_keeper', '--base', 'CNNBaseGfootball',
                              '--seed', '0', '--exp-name', 'test', '--cuda-deterministic', '--sync-every', '128',
                              '--use-linear-lr-decay', '--eval-every-step', '0'])
        subprocess.call([BIN, '../main.py', '--dump_run_id', '1', '--num-env-steps', '20500', '--dump_traj_flag',
                              '--eval-freq', '0', '--no-cuda', '--use-gae', '--num-processes', '8', '--num-actors', '8',
                              '--env-name', 'academy_3_vs_1_with_keeper', '--base', 'CNNBaseGfootball',
                              '--seed', '0', '--exp-name', 'test', '--cuda-deterministic', '--sync-every', '128',
                              '--use-linear-lr-decay', '--eval-every-step', '0'])
        self.prev_traj_runs0, self.prev_traj_runs1 = [], []
        self.num_actors = 8

        for actor_rank in range(self.num_actors):
            with open('/tmp/traj_run0_{}.pkl'.format(actor_rank), 'rb') as input:
                self.prev_traj_runs0.append(pickle.load(input))
            with open('/tmp/traj_run1_{}.pkl'.format(actor_rank), 'rb') as input:
                self.prev_traj_runs1.append(pickle.load(input))
            self.n_step = len(self.prev_traj_runs0[0]['obs'])



    def test_deter(self):
        for actor_rank in range(self.num_actors):
            self.assertEqual(self.prev_traj_runs0[actor_rank]['action'], self.prev_traj_runs1[actor_rank]['action'])
        for actor_rank in range(self.num_actors):
            self.assertEqual(self.prev_traj_runs0[actor_rank]['v'], self.prev_traj_runs1[actor_rank]['v'])
        for actor_rank in range(self.num_actors):
            for step in range(self.n_step):
                self.assertTrue(np.array_equal(self.prev_traj_runs0[actor_rank]['obs'][step], self.prev_traj_runs1[actor_rank]['obs'][step]))
        for actor_rank in range(self.num_actors):
            for step in range(self.n_step):
                self.assertTrue(np.array_equal(self.prev_traj_runs0[actor_rank]['action_logit'][step], self.prev_traj_runs1[actor_rank]['action_logit'][step]))






if __name__ == '__main__':
    unittest.main()