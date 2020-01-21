
class Episode:

    def __init__(self, support_loader, query_loader, base_task, task_id, n_classes):
        self.support_loader = support_loader
        self.query_loader = query_loader
        self.base_task = base_task
        self.task_id = task_id
        self.n_classes = n_classes



