import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

titlesize = 14
labelsize = 14
legendsize = 14

folders = ["20-260", "270-340", "360-400"]

# grafica AFD + strain
plt.figure("libre_AFDstra", figsize=[12,5])
plt.subplots_adjust(left=0.05, bottom=0.12, right=0.93, top=0.9, wspace=0.02, hspace=0)

# AFD
frames = []
for folder in folders:
    df = pd.read_csv(folder + "/csv/AFDa.csv")
    frames.append(df.copy())
AFDa = pd.concat(frames)
AFDa.to_csv("AFDaFINAL.csv", index=False)

ax1 = plt.subplot(121)
plt.title("(a) AFD$^{a}$ mode", fontsize=titlesize)
plt.errorbar(AFDa["temp"], AFDa["xrot"], yerr=AFDa["xrot_err"], label=r"AFD$_{x}^{a}$", marker ="<", linestyle="-", c="blue") 
plt.errorbar(AFDa["temp"], AFDa["yrot"], yerr=AFDa["yrot_err"], label=r"AFD$_{y}^{a}$", marker =">", linestyle="-", c="black") 
plt.errorbar(AFDa["temp"], AFDa["zrot"], yerr=AFDa["zrot_err"], label=r"AFD$_{z}^{a}$", marker ="^", linestyle="-", c="red")
plt.xlabel("$T$ (K)", fontsize = labelsize)
plt.ylabel(r"AFD rotation (deg)", fontsize = labelsize)
plt.ylim(0,8)
plt.legend(frameon = True, fontsize = legendsize)
plt.grid(True)


# strain 
frames = []
for folder in folders:
    df = pd.read_csv(folder + "/csv/strain.csv")
    frames.append(df.copy())
strains = pd.concat(frames)
strains.to_csv("strainFINAL.csv", index=False)

ax2 = plt.subplot(122)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.title("(b) Strain", fontsize=titlesize)
plt.errorbar(strains["temp"], strains["sx"]*100, yerr=strains["sx_err"]*100, label=r"$\eta_x$", marker ="<", linestyle="-",c="blue") 
plt.errorbar(strains["temp"], strains["sy"]*100, yerr=strains["sy_err"]*100, label=r"$\eta_y$", marker =">", linestyle="-",c="black") 
plt.errorbar(strains["temp"], strains["sz"]*100, yerr=strains["sz_err"]*100, label=r"$\eta_z$", marker ="^", linestyle="-",c="red")
plt.ylabel(r"$\eta$ (%)", fontsize = labelsize)
plt.xlabel("$T$ (K)", fontsize = labelsize)
plt.legend(frameon = True, fontsize = legendsize)
plt.grid(True)    


plt.savefig("libre_AFDstra.png")

#EXPANSION COEFFICIENT

def line(x, a, b):
    return a + b*x

newdf = strains.loc[strains["temp"] >= 320]

temps = np.array(newdf["temp"])
sx = np.array(newdf["sx"])
sx_err = np.array(newdf["sx_err"])
sy = np.array(newdf["sy"])
sy_err = np.array(newdf["sy_err"])
sz = np.array(newdf["sz"])
sz_err = np.array(newdf["sz_err"])


plt.figure("expansion-fits")

# x direction
popt, pcov = curve_fit(line, temps, sx, sigma=sx_err)
perr = np.sqrt(np.diag(pcov))

plt.errorbar(temps, sx, yerr=sx_err, label=r"$\eta_x$", marker ="<", linestyle="",c="blue")
plt.plot(temps, line(temps, popt[0], popt[1]), c="blue")

xx = popt[1]
xx_err = perr[1]

# y direction
popt, pcov = curve_fit(line, temps, sy, sigma=sy_err)
perr = np.sqrt(np.diag(pcov))

plt.errorbar(temps, sy, yerr=sy_err, label=r"$\eta_y$", marker =">", linestyle="",c="black")
plt.plot(temps, line(temps, popt[0], popt[1]), c="black")

xy = popt[1]
xy_err = perr[1]

# z direction
popt, pcov = curve_fit(line, temps, sz, sigma=sz_err)
perr = np.sqrt(np.diag(pcov))

plt.errorbar(temps, sz, yerr=sz_err, label=r"$\eta_z$", marker = "^", linestyle="",c="red")
plt.plot(temps, line(temps, popt[0], popt[1]), c="red")

xz = popt[1]
xz_err = perr[1]

#################################

print("expansion coefficients:")
print("x direction:", xx, "+-", xx_err, "K^-1")
print("y direction:", xy, "+-", xy_err, "K^-1")
print("z direction:", xz, "+-", xz_err, "K^-1")
print("average:", (xx+xy+xz)/3., "K^-1")



plt.show()
