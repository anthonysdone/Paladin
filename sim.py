class Reg:
    def __init__(self, init):
        self.val = init
        self.next = init
        self.init = init

    def tick(self):
        self.val = self.next

    def reset(self):
        self.val = self.init
        self.next = self.init
    
    def __repr__(self):
        return f"Reg({self.val})"

def task(func):
    def wrapper(*args, **kwargs):
        def task_fn():
            func(*args, **kwargs)
        return task_fn
    return wrapper

class Sim:
    def __init__(self):
        self.regs = []
        self.tasks = []
        self.cycle = 0
    
    def reg(self, init):
        r = Reg(init)
        self.regs.append(r)
        return r
    
    def add(self, task):
        self.tasks.append(task)
        return task
    
    def step(self):
        for task in self.tasks:
            task()

        for reg in self.regs:
            reg.tick()
        
        self.cycle += 1
    
    def run(self, cycles):
        for cycle in range(cycles):
            self.step()
    
    def reset(self):
        self.cycle = 0
        
        for reg in self.regs:
            reg.reset()