class ErasePseudoLabels:
    def __init__(self, pseudo_labels_path):
        self.pseudo_labels_path = pseudo_labels_path

    def __call__(self, session, epoch):
        with open(self.pseudo_labels_path, 'w') as f:
            pass


class RebuildIndex:
    def __call__(self, session, epoch):
        dataset = session.datasets["pseudo_labeled"]
        dataset.dataset.re_build()
