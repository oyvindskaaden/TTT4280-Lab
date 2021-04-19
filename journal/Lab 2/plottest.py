import csv
import matplotlib.pyplot as plt

header = []
data = []

filename = 'lpfilter.csv'
with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)

    header = next(csvreader)

    for datapoint in csvreader:

        values = [float(value) for value in datapoint]

        data.append(values)
print(header)
print(data[0])


#time = [p[0]*1000 for p in data]#[3928:]


time = [p[0] for p in data][:100]
ch1 = [p[1] for p in data][:100]
ch2 = [p[2] for p in data][:100]
#math2 = [p[3] for p in data]
print(time)
print(ch1)

print(header)
plt.semilogx(time, ch1, time, ch2)
plt.plot([time[0], time[-1]], [-3, -3])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude A [dB]")
plt.legend(["$V_{inn}$", "$V_{ut}$", "$-3dB$"])
plt.show()