import time


class Model:
    def predict(self, data, *args, **kwargs):
        print(f"Received {data}")
        print(f"Sleeping for 30 seconds")
        time.sleep(10)
        print("Done")
        return "done"
