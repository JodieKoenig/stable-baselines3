from multiprocessing import Pool, Process, Queue, Lock
import os
import time
import random


# def f(x):
#     return x*x


# def info(title):
#     print(title)
#     print('module name:', __name__)
#     print('parent process:', os.getppid())
#     print('process id:', os.getpid())


# def f(name):
#     info('function f')
#     print('hello', name)


# def f(l, i):
#     l.acquire()
#     try:
#         print('hello world', i)
#     finally:
#         l.release()


def qu(agent, i, queue):
    while i % 10:
        i = i+1
    queue.put([agent, i, None, 'hello'])


if __name__ == '__main__':
    q = Queue()
    continue_checking = 1
    num_agents = 4
    processes = []

    for number in range(num_agents):
        i = random.randint(0, 10)
        process = Process(target=qu, args=(number, i, q))
        processes.append(process)

    for number in range(num_agents):
        processes[number].start()

    while continue_checking:
        info = q.get()
        num_process = info[0]
        current_number = int(info[1])
        if info[1] == 100:
            break
        print("process number:", num_process, "and count:", current_number)
        current_number = int(current_number) + 1
        process = Process(target=qu, args=(num_process, current_number, q))
        processes[num_process] = process
        processes[num_process].start()

    # lock = Lock()
    #
    # for num in range(10):
    #     Process(target=f, args=(lock, num)).start()

    # info('main line')
    # p1 = Process(target=f, args=('bob',))
    # p2 = Process(target=f, args=('james',))
    # p1.start()
    # p2.start()

    # with Pool(5) as p:
    #     print(p.map(f, [1, 2, 3]))

    # # start 4 worker processes
    # with Pool(processes=4) as pool:
    #
    #     # print "[0, 1, 4,..., 81]"
    #     print(pool.map(f, range(10)))
    #
    #     # print same numbers in arbitrary order
    #     for i in pool.imap_unordered(f, range(10)):
    #         print(i)
    #
    #     # evaluate "f(20)" asynchronously
    #     res = pool.apply_async(f, (20,))      # runs in *only* one process
    #     print(res.get(timeout=1))             # prints "400"
    #
    #     # evaluate "os.getpid()" asynchronously
    #     res = pool.apply_async(os.getpid, ())   # runs in *only* one process
    #     print(res.get(timeout=1))               # prints the PID of that process
    #
    #     # launching multiple evaluations asynchronously *may* use more processes
    #     multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
    #     print([res.get(timeout=1) for res in multiple_results])
    #
    #     # make a single worker sleep for 10 secs
    #     res = pool.apply_async(time.sleep, (10,))
    #     try:
    #         print(res.get(timeout=1))
    #     except TimeoutError:
    #         print("We lacked patience and got a multiprocessing.TimeoutError")
    #
    #     print("For the moment, the pool remains available for more work")
    #
    # # exiting the 'with'-block has stopped the pool
    # print("Now the pool is closed and no longer available")
    # # cpu_count()

# Compare single proc vs multiple procs execution for cpu bound operation
#
# """
# Typical Result:
#   Starting 1000000 cycles of cpu-only processing
#   Sequential run time: 9.24 seconds
#   4 procs Parallel - run time: 2.59 seconds
#   2 procs Parallel twice - run time: 4.76 seconds
# """
# import time
# import multiprocess as mp
# from torch import multiprocessing as multi
#
# # one million
# cycles = 1000 * 1000
#
# def t():
#     for x in range(cycles):
#         fdivision = cycles / 2.0
#         fcomparison = (x > fdivision)
#         faddition = fdivision + 1.0
#         fsubtract = fdivision - 2.0
#         fmultiply = fdivision * 2.0
#
# if __name__ == '__main__':
#     print("  Starting {} cycles of cpu-only processing".format(cycles))
#     start_time = time.time()
#     t()
#     t()
#     t()
#     t()
#     print("  Sequential run time: %.2f seconds" % (time.time() - start_time))
#
#     # four procs
#     start_time = time.time()
#     p1 = mp.Process(target=t)
#     p2 = mp.Process(target=t)
#     p3 = mp.Process(target=t)
#     p4 = mp.Process(target=t)
#     p1.start()
#     p2.start()
#     p3.start()
#     p4.start()
#     p1.join()
#     p2.join()
#     p3.join()
#     p4.join()
#     print("  4 procs Parallel - run time: %.2f seconds" % (time.time() - start_time))
#
#     # two procs
#     start_time = time.time()
#     p1 = mp.Process(target=t)
#     p2 = mp.Process(target=t)
#     p1.start()
#     p2.start()
#     p1.join()
#     p2.join()
#     p3 = mp.Process(target=t)
#     p4 = mp.Process(target=t)
#     p3.start()
#     p4.start()
#     p3.join()
#     p4.join()
#     print("  2 procs Parallel twice - run time: %.2f seconds" % (time.time() - start_time))
# #
# # while (not finished)
# #     get()