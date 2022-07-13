class LogWriter:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "w")
        self.f.close()

    def __call__(self, *args, **kwargs):
        self.f = open(self.path, "a")
        sep, end = " ", "\n"
        for k, v in kwargs.items():
            if k == "sep": 
                sep = v
            elif k == "end":
                end = v
        for arg in args:
            if not isinstance(arg, str):
                arg = str(arg)
            self.f.write(arg)
            self.f.write(sep)
            print(arg, end = sep)
        self.f.write(end)
        print("", end = end)
        self.f.close()
