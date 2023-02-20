import time

import torch
from tqdm import tqdm

from utils import flat_accuracy, format_time



class IterResult:
    def __init__(self, datasize):
        self.start = time.time()
        self.payload = {}
        self.datasize = datasize

    def mark(self, **measures):
        report = {k: measures[k] / self.datasize for k in measures}
        report["time_used"] = format_time(time.time() - self.start)
        return report

class Trainer:
    def __init__(self, model, optimzer, scheduler, train_dl, val_dl, epochs=1):
        self.model = model
        self.optimzer = optimzer
        self.scheduler = scheduler
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.epochs = epochs
        self.total_steps = len(train_dl) * self.epochs

    def train(self):
        training_stats = []

        for epoch_i in range(0, self.epochs):
            print("")
            print(
                '======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print("\r\n", "Training...")
            self.model.train()
            train_report = self.nn_fnb_propagation(start=time.time())
            print("\r\n", "  Average training loss: {0:.2f}".format(
                train_report["avg_loss"]))
            print("  Training epcoh took: {:}".format(
                train_report["time_used"]))
            print("\r\n", "Running validation...")
            self.model.eval()

            val_report = self.nn_validation()
            print("  Accuracy: {0:.2f}".format(val_report["avg_accy"]))
            print("  Validation Loss: {0:.2f}".format(val_report["avg_loss"]))
            print("  Validation took: {:}".format(val_report["time_used"]))

            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': train_report["avg_loss"],
                    'Valid. Loss': val_report["avg_loss"],
                    'Valid. Accur.': val_report["avg_accy"],
                    'Training Time': train_report["time_used"],
                    'Validation Time': val_report["time_used"]
                }
            )

    def nn_fnb_propagation(self, start):
        o = self.optimzer
        s = self.scheduler
        m = self.model
        tdl = self.train_dl
        propagation_loss = 0

        iter_result = IterResult(len(tdl))

        for step, batch in tqdm(enumerate(tdl), total=len(tdl)):
            b_input_ids = batch[0].to('cpu')
            b_input_mask = batch[1].to('cpu')
            b_labels = batch[2].to('cpu')

            m.zero_grad()
            outputs = m(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
            propagation_loss = propagation_loss + loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o.step()
            s.step()

            # if step and step % 4 == 0:
            #     m.eval()
            #     val_report = self.nn_validation()
            #     print("\r\n", " Accuracy: {0:.2f}".format(val_report["avg_accy"]))
            #     m.train()

        return iter_result.mark(avg_loss=propagation_loss)

    def nn_validation(self):
        m = self.model
        vdl = self.val_dl
        eval_loss = 0
        eval_accy = 0

        iter_result = IterResult(len(vdl))

        for batch in vdl:
            b_input_ids = batch[0].to('cpu')
            b_input_mask = batch[1].to('cpu')
            b_labels = batch[2].to('cpu')

            with torch.no_grad():
                outputs = m(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels
                            )
                loss, logits = outputs.loss, outputs.logits

            eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            eval_accy += flat_accuracy(logits, label_ids)

        return iter_result.mark(avg_loss=eval_loss, avg_accy=eval_accy)
