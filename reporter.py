import time, math


class Reporter(object):
    def __init__(self, log_frequency, logger):
        self.log_frequency = log_frequency
        self.logger = logger

        self.global_step = 0
        self.lr = 0
        self.init_stats()

    def init_stats(self):
        self.step_time = 0.0

        self.resp_loss = 0.0

        self.resp_correct = 0.0

        self.resp_count = 0.0

    def step(self, start_time, lr, step_outputs, force_info=False, is_train=True):
        self.global_step += 1
        self.step_time += (time.time() - start_time)

        self.resp_loss += step_outputs["loss"]
        self.resp_correct += step_outputs["correct"]
        self.resp_count += step_outputs["count"]

        if is_train:
            self.lr = lr

            if self.global_step % self.log_frequency == 0:
                self.info_stats("train", self.global_step)

    def info_stats(self, data_type, global_step=None):
        avg_step_time = self.step_time / self.log_frequency

        if data_type == "train":
            common_info = "step {0:d}; step-time {1:.2f}s; lr {2:.2e};".format(
                global_step, avg_step_time, self.lr)
        else:
            common_info = "[Validation]"

        resp_ppl = math.exp(self.resp_loss / self.resp_count)
        resp_acc = (self.resp_correct / self.resp_count) * 100

        resp_info = "[resp] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
            self.resp_loss, resp_ppl, resp_acc)

        self.logger.info(
            " ".join([common_info, resp_info]))

        self.init_stats()