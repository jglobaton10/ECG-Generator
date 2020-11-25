##Modelo de ecuaciones diferenciales
import numpy as np
import matplotlib.pyplot as plt

#--------------------------Parametros iniciales del modelo -------------------------------------------------------------


Frecuencia_cardiaca = 80
Numero_latidos = 10   #Duracion de la señal
T0 = 0
Tf = Numero_latidos
Frecuancia_muestreo = 360
h = 1/Frecuancia_muestreo
T = np.arange(T0, Tf + h, h)
frecuancia_cardiaca_media = 60/Frecuencia_cardiaca
#Tiempo aleatorio
Trr = np.random.normal(frecuancia_cardiaca_media, 0.07*frecuancia_cardiaca_media,np.size(T))
factor_ruido = np.sin(T)



#-------------------------------Parametros de la forma de la onda-------------------------------------------------------
thetai = [-(1/3)*np.pi ,-(1/12)*np.pi ,0, (1/12)*np.pi, (1/2)*np.pi]
ai = [1.2, -5,30,-7.5,0.75]
bi = [0.25,0.1,0.1,0.1,0.4]
"""
--------------------------Ecuaciones del modelo ------------------------------------------------------------------------
"""

#------------------------------------Ecuaciones Euler Fordward----------------------------------------------------------
def F1(y1,y2,Trr):
    a = 1 - np.sqrt(y1**2 + y2**2)
    w = 2*np.pi/Trr
    return a * y1 - w * y2
def F2(y1,y2,Trr):
    a = 1 - np.sqrt(y1**2 + y2**2)
    w = 2*np.pi/Trr
    return a * y2 + w * y1
def F3(ai, bi,y1,y2,y3, thetai,t):
    theta = np.arctan2(y1, y2)

    dthetai = np.fmod(theta - thetai, 2*np.pi) #Diferencial de thetai
    f2 = 0.25 #Frecuencia respiratoria
    zbase = 0.005 * np.sin(2 * np.pi * f2 * t)
    return  np.sum(ai * dthetai * np.exp(-0.5 * (dthetai / bi) ** 2)) - (y3 - zbase)
#-----------------------------------Ecuaciones Euler backwards----------------------------------------------------------
def  F1Back(y1,y2,Trr,h):
    a = 1 - np.sqrt(y1 ** 2 + y2 ** 2)
    w = 2 * np.pi / Trr
    return (y1-h*w*y2)/(1-h*a)
def F2Back(y1,y2,Trr,h):
    a = 1 - np.sqrt(y1 ** 2 + y2 ** 2)
    w = 2 * np.pi / Trr
    return (y2+h*w*y1)/(1-h*a)
def F3Back(ai, bi,y1,y2,y3, thetai,t,h):
    theta = np.arctan2(y1, y2)

    dthetai = np.fmod(theta - thetai, 2 * np.pi)  # Diferencial de thetai
    f2 = 0.25  # Frecuencia respiratoria
    zbase = 0.005 * np.sin(2 * np.pi * f2 * t)
    return (y3 +h*np.sum(ai * dthetai * np.exp(-0.5 * (dthetai / bi) ** 2))  + h*zbase)/(1+h)
#-------------------------------------------Ecuaciones Euler Modificado-------------------------------------------------
def  F1Modificado(y1,y2,Trr,h):
    a = 1 - np.sqrt(y1 ** 2 + y2 ** 2)
    w = 2 * np.pi / Trr
    return (2*y1-h*w*y2)/(2-h*a)
def F2Modificado(y1,y2,Trr,h):
    a = 1 - np.sqrt(y1 ** 2 + y2 ** 2)
    w = 2 * np.pi / Trr
    return (2*y2+h*w*y1)/(2-h*a)
def F3Modificado(ai, bi,y1,y2,y3,thetai,t,h):
    theta = np.arctan2(y1, y2)
    dthetai = np.fmod(theta - thetai, 2 * np.pi)  # Diferencial de thetai
    f2 = 0.25  # Frecuencia respiratoria
    zbase = 0.005 * np.sin(2 * np.pi * f2 * t)
    return (y3 +h*np.sum(ai * dthetai * np.exp(-0.5 * (dthetai / bi) ** 2))  + h*zbase)/(2+h)


Y1 = 0
Y2 = 1
Y3 = 0.015
#Euler Forward
Y1Eulerfor = np.zeros(len(T))
Y2Eulerfor = np.zeros(len(T))
Y3Eulerfor = np.zeros(len(T))
Y1Eulerfor[0] = Y1
Y2Eulerfor[0] = Y2
Y3Eulerfor[0] = Y3
#Euler Backwards
Y1EulerBack = np.zeros(len(T))
Y2EulerBack = np.zeros(len(T))
Y3EulerBack = np.zeros(len(T))
Y1EulerBack[0] = Y1
Y2EulerBack[0] = Y2
Y3EulerBack[0] = Y3
#Rugen Kutta 2
Y1Rugen = np.zeros(len(T))
Y2Rugen = np.zeros(len(T))
Y3Rugen = np.zeros(len(T))
Y1Rugen[0] = Y1
Y2Rugen[0] = Y2
Y3Rugen[0] = Y3
#Rugen Kutta 4
Y1Rugen4= np.zeros(len(T))
Y2Rugen4 = np.zeros(len(T))
Y3Rugen4 = np.zeros(len(T))
Y1Rugen4[0] = Y1
Y2Rugen4[0] = Y2
Y3Rugen4[0] = Y3
#Euler Backwards
Y1EulerModificado = np.zeros(len(T))
Y2EulerModificado= np.zeros(len(T))
Y3EulerModificado = np.zeros(len(T))
Y1EulerModificado[0] = Y1
Y2EulerModificado[0] = Y2
Y3EulerModificado[0] = Y3


for iter in range(1, len(T)):
    #Euler Forward
    Y1Eulerfor[iter] = Y1Eulerfor[iter -1] + h * F1(Y1Eulerfor[iter -1],Y2Eulerfor[iter -1],Trr[iter -1])
    Y2Eulerfor[iter] = Y2Eulerfor[iter - 1] + h * F2(Y1Eulerfor[iter - 1], Y2Eulerfor[iter - 1], Trr[iter -1])
    Y3Eulerfor[iter] = Y3Eulerfor[iter - 1] + h * F3(ai,bi,Y1Eulerfor[iter -1],Y2Eulerfor[iter -1],Y3Eulerfor[iter -1],thetai,T[iter -1])
    #Euler Backwards
    Y1EulerBack [iter] =  F1Back(Y1EulerBack[iter -1],Y2EulerBack[iter -1],Trr[iter -1],h)
    Y2EulerBack [iter] =  F2Back(Y1EulerBack[iter - 1], Y2EulerBack[iter - 1], Trr[iter -1],h)
    Y3EulerBack [iter] =  F3Back(ai,bi,Y1EulerBack[iter -1],Y2EulerBack[iter -1], Y3EulerBack [iter-1],thetai,T[iter -1],h)
    #Rugen kUtta 2
    Y1k1 = F1(Y1Rugen[iter-1],Y2Rugen[iter-1],Trr[iter -1])
    Y1k2 = F1(Y1Rugen[iter-1] + Y1k1 *h, Y2Rugen[iter-1]+ Y1k1 *h , Trr[iter -1])
    Y1Rugen[iter] = Y1Rugen[iter-1] + (h/2.0)*(Y1k1 + Y1k2)

    Y2k1 = F2(Y1Rugen[iter - 1],  Y2Rugen[iter - 1], Trr[iter -1])
    Y2k2 = F2(Y1Rugen[iter - 1] + Y2k1 * h, Y2Rugen[iter - 1] + Y2k1 * h, Trr[iter -1])
    Y2Rugen[iter] = Y2Rugen[iter - 1] + (h / 2.0) * (Y2k1 + Y2k2)

    Y3k1 = F3(ai,bi,Y1Rugen[iter - 1], Y2Rugen[iter - 1],Y3Rugen[iter - 1], thetai, T[iter-1])
    Y3k2 = F3(ai,bi,Y1Rugen[iter - 1] + Y3k1 * h , Y2Rugen[iter - 1] + Y3k1 * h , Y3Rugen[iter - 1] + Y3k1 * h,thetai,T[iter-1])
    Y3Rugen[iter] = Y3Rugen[iter - 1] + (h / 2.0) * (Y3k1 + Y3k2)

    #Rugen Kutta 4

    Y14k1 = F1(Y1Rugen4[iter - 1],Y2Rugen4[iter - 1],Trr[iter - 1])
    Y14k2 = F1(Y1Rugen4[iter - 1] + 0.5 * Y14k1 * h,Y2Rugen4[iter - 1] + 0.5 * Y14k1 * h,Trr[iter - 1] + 0.5 * h)
    Y14k3 = F1(Y1Rugen4[iter - 1] + 0.5 * Y14k2 * h,Y2Rugen4[iter - 1] + 0.5 * Y14k2 * h,Trr[iter - 1] + 0.5 * h)
    Y14k4 = F1(Y1Rugen4[iter - 1] + Y14k3 * h,Y2Rugen4[iter - 1] + Y14k3 * h,T[iter - 1] + h)
    Y1Rugen4[iter] = Y1Rugen4[iter - 1] + (h / 6.0) * (Y14k1 + Y14k2 * 2 + 2 * Y14k3 + Y14k4)

    Y24k1 = F2(Y1Rugen4[iter - 1], Y2Rugen4[iter - 1], Trr[iter - 1])
    Y24k2 = F2(Y1Rugen4[iter - 1] + 0.5 * Y24k1 * h, Y2Rugen4[iter - 1] + 0.5 * Y24k1 * h, Trr[iter - 1] + 0.5 * h)
    Y24k3 = F2(Y1Rugen4[iter - 1] + 0.5 * Y24k2 * h, Y2Rugen4[iter - 1] + 0.5 * Y24k2 * h, Trr[iter - 1] + 0.5 * h)
    Y24k4 = F2(Y1Rugen4[iter - 1] + Y24k3 * h, Y2Rugen4[iter - 1] + Y24k3 * h,T[iter - 1] + h)
    Y2Rugen4[iter] = Y2Rugen4[iter - 1] + (h / 6.0) * (Y24k1 + Y24k2 * 2 + 2 * Y24k3 + Y24k4)

    Y34k1 = F3(ai,bi,Y1Rugen4[iter - 1], Y2Rugen4[iter - 1], Y3Rugen4[iter - 1], thetai, T[iter-1])
    Y34k2 = F3(ai,bi,Y1Rugen4[iter - 1] + 0.5 * Y34k1 * h, Y2Rugen4[iter - 1] + 0.5 * Y34k1 * h,Y3Rugen4[iter - 1] + 0.5 * Y34k1 * h,thetai, T[iter-1] + 0.5 * h)
    Y34k3 = F3(ai,bi,Y1Rugen4[iter - 1] + 0.5 * Y34k2 * h, Y2Rugen4[iter - 1] + 0.5 * Y34k2 * h,Y3Rugen4[iter - 1] + 0.5 * Y34k2 * h,thetai, T[iter-1] + 0.5 * h)
    Y34k4 = F3(ai,bi,Y1Rugen4[iter - 1] + Y34k3 * h, Y2Rugen4[iter - 1] + Y34k3 * h, Y3Rugen4[iter - 1] + Y34k3 * h,thetai,T[iter-1] + h)
    Y3Rugen4[iter] = Y3Rugen4[iter - 1] + (h / 6.0) * (Y34k1 + Y34k2 * 2 + 2 * Y34k3 + Y34k4)

    #Euler Modificado
    Y1EulerModificado[iter] = F1Modificado(Y1EulerModificado[iter - 1], Y2EulerModificado[iter - 1], Trr[iter - 1], h)
    Y2EulerModificado[iter] = F2Modificado(Y1EulerModificado[iter - 1], Y2EulerModificado[iter - 1], Trr[iter - 1], h)
    Y3EulerModificado[iter] = F3Modificado(ai, bi, Y1EulerModificado[iter - 1], Y2EulerModificado[iter - 1], Y3EulerModificado[iter - 1], thetai,T[iter - 1], h)
#-------------------------------------Metodo auxiliares ----------------------------------------------------------------
#--------------------------------------Metodo de exportacion------------------------------------------------------------
import struct as st
var_exportacion = Y3Eulerfor
def Exportar():
    fHdl = open("ArchivoProyecto.bin", "bw")
    var2 = st.pack(len(T) * 'd', *var_exportacion)  # Escribe un numero a la f es float 32 bits
    fHdl.write(var2)
    fHd2 = open("ArchivoProyecto2.bin", "bw")
    var3 = st.pack(len(T) * 'd', *T)  # Escribe un numero a la f es float 32 bits
    fHd2.write(var3)
    # var3 = st.unpack('d'*int((len( fHdl.read())/8)) , fHdl.read())
    # fHdl.close()
    print("El archivo se guardo correctamente")


#----------------------------------Metodo de carga----------------------------------------------------------------------
def Cargar():
    fHdl = open("ArchivoProyecto.bin", "br")
    var2 = fHdl.read()
    var3 = st.unpack('d' * int((len(var2) / 8)), var2)
    fHdl.close()

    fHd2 = open("ArchivoProyecto2.bin", "br")
    var22 = fHd2.read()
    var32 = st.unpack('d' * int((len(var22) / 8)), var22)
    fHd2.close()
    plt.figure()
    plt.plot(var32,var3,"black")
    plt.xlabel("tiempo")
    plt.ylabel("Voltaje")


#--------------------------------Funcion para encontrar picos-----------------------------------------------------------

from  scipy.signal import  find_peaks

#time = np.arange(len(Y3EulerModificado))/Frecuancia_muestreo

def Heart_Rate():
    global var_exportacion
    peaks, properties= find_peaks(var_exportacion, height=0.011 )  # Permite detectar las ondas que corresponden a los latidos
    #Time_ecg = T[peaks]
    #Time_ecg = T[1:]
    taco = np.diff(T[peaks])
    tacobpm = 60 / taco
    resultado_HR.set(str(np.mean(tacobpm)))  # HR
#print(Heart_Rate(Y3EulerModificado,0.01 ))
#-----------------------------------Funcion de modificacion de parametros-----------------------------------------------
def repetirFunciones():
    global Y1
    global Y2
    global Y3
    global Y1Eulerfor
    global Y2Eulerfor
    global Y3Eulerfor
    global Y1EulerBack
    global Y2EulerBack
    global Y3EulerBack
    global Y1EulerModificado
    global Y2EulerModificado
    global Y3EulerModificado
    global Y1Rugen
    global Y2Rugen
    global Y3Rugen
    global Y1Rugen4
    global Y2Rugen4
    global Y3Rugen4
    Y1 = 0
    Y2 = 1
    Y3 = 0.015
    # Euler Forward
    Y1Eulerfor = np.zeros(len(T))
    Y2Eulerfor = np.zeros(len(T))
    Y3Eulerfor = np.zeros(len(T))
    Y1Eulerfor[0] = Y1
    Y2Eulerfor[0] = Y2
    Y3Eulerfor[0] = Y3
    # Euler Backwards
    Y1EulerBack = np.zeros(len(T))
    Y2EulerBack = np.zeros(len(T))
    Y3EulerBack = np.zeros(len(T))
    Y1EulerBack[0] = Y1
    Y2EulerBack[0] = Y2
    Y3EulerBack[0] = Y3
    # Rugen Kutta 2
    Y1Rugen = np.zeros(len(T))
    Y2Rugen = np.zeros(len(T))
    Y3Rugen = np.zeros(len(T))
    Y1Rugen[0] = Y1
    Y2Rugen[0] = Y2
    Y3Rugen[0] = Y3
    # Rugen Kutta 4
    Y1Rugen4 = np.zeros(len(T))
    Y2Rugen4 = np.zeros(len(T))
    Y3Rugen4 = np.zeros(len(T))
    Y1Rugen4[0] = Y1
    Y2Rugen4[0] = Y2
    Y3Rugen4[0] = Y3
    # Euler Backwards
    Y1EulerModificado = np.zeros(len(T))
    Y2EulerModificado = np.zeros(len(T))
    Y3EulerModificado = np.zeros(len(T))
    Y1EulerModificado[0] = Y1
    Y2EulerModificado[0] = Y2
    Y3EulerModificado[0] = Y3

    for iter in range(1, len(T)):
        # Euler Forward
        Y1Eulerfor[iter] = Y1Eulerfor[iter - 1] + h * F1(Y1Eulerfor[iter - 1], Y2Eulerfor[iter - 1], Trr[iter - 1])
        Y2Eulerfor[iter] = Y2Eulerfor[iter - 1] + h * F2(Y1Eulerfor[iter - 1], Y2Eulerfor[iter - 1], Trr[iter - 1])
        Y3Eulerfor[iter] = Y3Eulerfor[iter - 1] + h * F3(ai, bi, Y1Eulerfor[iter - 1], Y2Eulerfor[iter - 1],
                                                         Y3Eulerfor[iter - 1], thetai, T[iter - 1])
        # Euler Backwards
        Y1EulerBack[iter] = F1Back(Y1EulerBack[iter - 1], Y2EulerBack[iter - 1], Trr[iter - 1], h)
        Y2EulerBack[iter] = F2Back(Y1EulerBack[iter - 1], Y2EulerBack[iter - 1], Trr[iter - 1], h)
        Y3EulerBack[iter] = F3Back(ai, bi, Y1EulerBack[iter - 1], Y2EulerBack[iter - 1], Y3EulerBack[iter - 1], thetai,
                                   T[iter - 1], h)
        # Rugen kUtta 2
        Y1k1 = F1(Y1Rugen[iter - 1], Y2Rugen[iter - 1], Trr[iter - 1])
        Y1k2 = F1(Y1Rugen[iter - 1] + Y1k1 * h, Y2Rugen[iter - 1] + Y1k1 * h, Trr[iter - 1])
        Y1Rugen[iter] = Y1Rugen[iter - 1] + (h / 2.0) * (Y1k1 + Y1k2)

        Y2k1 = F2(Y1Rugen[iter - 1], Y2Rugen[iter - 1], Trr[iter - 1])
        Y2k2 = F2(Y1Rugen[iter - 1] + Y2k1 * h, Y2Rugen[iter - 1] + Y2k1 * h, Trr[iter - 1])
        Y2Rugen[iter] = Y2Rugen[iter - 1] + (h / 2.0) * (Y2k1 + Y2k2)

        Y3k1 = F3(ai, bi, Y1Rugen[iter - 1], Y2Rugen[iter - 1], Y3Rugen[iter - 1], thetai, T[iter - 1])
        Y3k2 = F3(ai, bi, Y1Rugen[iter - 1] + Y3k1 * h, Y2Rugen[iter - 1] + Y3k1 * h, Y3Rugen[iter - 1] + Y3k1 * h,
                  thetai, T[iter - 1])
        Y3Rugen[iter] = Y3Rugen[iter - 1] + (h / 2.0) * (Y3k1 + Y3k2)

        # Rugen Kutta 4

        Y14k1 = F1(Y1Rugen4[iter - 1], Y2Rugen4[iter - 1], Trr[iter - 1])
        Y14k2 = F1(Y1Rugen4[iter - 1] + 0.5 * Y14k1 * h, Y2Rugen4[iter - 1] + 0.5 * Y14k1 * h, Trr[iter - 1] + 0.5 * h)
        Y14k3 = F1(Y1Rugen4[iter - 1] + 0.5 * Y14k2 * h, Y2Rugen4[iter - 1] + 0.5 * Y14k2 * h, Trr[iter - 1] + 0.5 * h)
        Y14k4 = F1(Y1Rugen4[iter - 1] + Y14k3 * h, Y2Rugen4[iter - 1] + Y14k3 * h, T[iter - 1] + h)
        Y1Rugen4[iter] = Y1Rugen4[iter - 1] + (h / 6.0) * (Y14k1 + Y14k2 * 2 + 2 * Y14k3 + Y14k4)

        Y24k1 = F2(Y1Rugen4[iter - 1], Y2Rugen4[iter - 1], Trr[iter - 1])
        Y24k2 = F2(Y1Rugen4[iter - 1] + 0.5 * Y24k1 * h, Y2Rugen4[iter - 1] + 0.5 * Y24k1 * h, Trr[iter - 1] + 0.5 * h)
        Y24k3 = F2(Y1Rugen4[iter - 1] + 0.5 * Y24k2 * h, Y2Rugen4[iter - 1] + 0.5 * Y24k2 * h, Trr[iter - 1] + 0.5 * h)
        Y24k4 = F2(Y1Rugen4[iter - 1] + Y24k3 * h, Y2Rugen4[iter - 1] + Y24k3 * h, T[iter - 1] + h)
        Y2Rugen4[iter] = Y2Rugen4[iter - 1] + (h / 6.0) * (Y24k1 + Y24k2 * 2 + 2 * Y24k3 + Y24k4)

        Y34k1 = F3(ai, bi, Y1Rugen4[iter - 1], Y2Rugen4[iter - 1], Y3Rugen4[iter - 1], thetai, T[iter - 1])
        Y34k2 = F3(ai, bi, Y1Rugen4[iter - 1] + 0.5 * Y34k1 * h, Y2Rugen4[iter - 1] + 0.5 * Y34k1 * h,
                   Y3Rugen4[iter - 1] + 0.5 * Y34k1 * h, thetai, T[iter - 1] + 0.5 * h)
        Y34k3 = F3(ai, bi, Y1Rugen4[iter - 1] + 0.5 * Y34k2 * h, Y2Rugen4[iter - 1] + 0.5 * Y34k2 * h,
                   Y3Rugen4[iter - 1] + 0.5 * Y34k2 * h, thetai, T[iter - 1] + 0.5 * h)
        Y34k4 = F3(ai, bi, Y1Rugen4[iter - 1] + Y34k3 * h, Y2Rugen4[iter - 1] + Y34k3 * h,
                   Y3Rugen4[iter - 1] + Y34k3 * h, thetai, T[iter - 1] + h)
        Y3Rugen4[iter] = Y3Rugen4[iter - 1] + (h / 6.0) * (Y34k1 + Y34k2 * 2 + 2 * Y34k3 + Y34k4)

        # Euler Modificado
        Y1EulerModificado[iter] = F1Modificado(Y1EulerModificado[iter - 1], Y2EulerModificado[iter - 1], Trr[iter - 1],
                                               h)
        Y2EulerModificado[iter] = F2Modificado(Y1EulerModificado[iter - 1], Y2EulerModificado[iter - 1], Trr[iter - 1],
                                               h)
        Y3EulerModificado[iter] = F3Modificado(ai, bi, Y1EulerModificado[iter - 1], Y2EulerModificado[iter - 1],
                                               Y3EulerModificado[iter - 1], thetai, T[iter - 1], h)
def cambiar_parametros():
    global Frecuencia_cardiaca
    Frecuencia_cardiaca = Frecuancia_Cardiaca.get()
    global Numero_latidos
    Numero_latidos = N_latidos.get()
    global Frecuancia_muestreo
    Frecuancia_muestreo = Frecuancia_Muestreo.get()
    global h
    h = 1/Frecuancia_muestreo
    global Tf
    Tf = Numero_latidos
    global T
    T= np.arange(T0, Tf + h, h)
    global frecuancia_cardiaca_media
    frecuancia_cardiaca_media = 60/Frecuencia_cardiaca
    repetirFunciones()

def cambiarAB():
    ai[0] = Pa.get()
    ai[1] = Qa.get()
    ai[2] = Ra.get()
    ai[3] = Sa.get()
    ai[4] = Ta.get()
    bi[0] = Pb.get()
    bi[1] = Qb.get()
    bi[2] = Rb.get()
    bi[3] = Rb.get()
    bi[4] = Tb.get()

    repetirFunciones()
#----------------------------------Funcion para graficar----------------------------------------------------------------


def GraficadorSelec():
    global var_exportacion

    if val_eulerFor.get() == 1:
        var_exportacion = Y3Eulerfor
        print("For")
        return T,Y3Eulerfor, 'r'  # subplot(filas, columnas, item)
    if val_eulerBack.get() == 1:
        var_exportacion= Y3EulerBack
        print("Back")
        return T,Y3EulerBack, 'b'
    if val_eulerMod.get() == 1:
        var_exportacion= Y3EulerModificado
        print("Mod")
        return T,Y3EulerModificado, 'yellow'
    if val_Ruggen2.get() == 1:
        var_exportacion = Y3Rugen
        print("Ruggen2")
        return T,Y3Rugen, 'black'
    if val_Ruggen4.get() == 1:
        var_exportacion=Y3Rugen4
        print("Ruggen4")
        return T,Y3Rugen4, 'g'


def Graficador():
    vT, result, color = GraficadorSelec()
    fig = plt.Figure(figsize=(5, 3), dpi=80)
    fig.add_subplot(111).plot(vT, result , color)
    #plt.ylabel("Voltaje")
    #plt.xlabel("Tiempo")
    plt.close()
    plt.style.use('seaborn-darkgrid')
    Plot = FigureCanvasTkAgg(fig, master=window)
    Plot.draw()
    Plot.get_tk_widget().place(x=40, y=170)
    """
    toolbar = NavigationToolbar2Tk(Plot, window)
    toolbar.update()
    def on_key_press(event):
        key_press_handler(event, Plot, toolbar)

    Plot.mpl_connect("key_press_event", on_key_press)
    """







#-----------------------------------------------------Interfaz---------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from PIL import ImageTk ,Image
from matplotlib import style



#Se configuran los parametros inciales de la ventana
window = tk.Tk()                    # Frame principal
window.geometry('800x670')          # Tamaño de la ventana
window.title('Proyecto final ECG')  #Título de la ventan
window.config(cursor="arrow")       #Tipo de cursor

#Creacion de los paneles derechos
#Creación del panel de los parametros----------------------------------------------------------------------------
Panel_parametros = tk.Frame(master = window) #Se asocia a la ventana principal
Panel_parametros.place(x=480, y=20)
Panel_parametros.config(bg="#FFFFFF", width=300, height=300, relief=tk.SOLID, bd=3)

#labels del panel
lbl_titulo1 = tk.Label(master=Panel_parametros, bg = 'white',fg = '#ffa72b',font=('Times', 15, 'bold italic'), text="                                                ").grid(row=0, column=0,columnspan = 2, rowspan = 1 )
lbl_titulo2 = tk.Label(master=Panel_parametros, bg = 'white',fg = '#ffa72b',font=('Times', 15, 'bold italic'), text="                     Parámetros                 ").grid(row=1, column=0,columnspan = 2, rowspan = 1 )
#Labels de izquiera
lbl_Frecuancia  = tk.Label(master=Panel_parametros, fg = 'white',bg = '#ffa72b',font=('Times', 12),relief='groove', text=" Frecuancia  Cardiaca ",width = 15).grid(row=2, column=0 )
lbl_N_latidos  = tk.Label(master=Panel_parametros, fg = 'white',bg = '#ffa72b',font=('Times', 12),relief='groove', text=" # de latidos         ",width = 15).grid(row=3, column=0 )
lbl_Frecuancia_Muestreo  = tk.Label(master=Panel_parametros, fg = 'white',bg = '#ffa72b',font=('Times', 12),relief='groove', text=" Frecuancia  Muestreo ",width = 15).grid(row=4, column=0 )
lbl_Factor_Ruido  = tk.Label(master=Panel_parametros, fg = 'white',bg = '#ffa72b',font=('Times', 12),relief='groove', text=" Factor de Ruido      ",width = 15).grid(row=5, column=0 )

#Text boxes


Frecuancia_Cardiaca = tk.IntVar()
tb_FC = tk.Entry(master=Panel_parametros, textvariable=Frecuancia_Cardiaca,width = 10,relief='solid').grid(row=2, column=1)
N_latidos = tk.IntVar()
tb_NL= tk.Entry(master=Panel_parametros, textvariable=N_latidos,width = 10,relief='solid').grid(row=3, column=1)
Frecuancia_Muestreo = tk.IntVar()
tb_FM = tk.Entry(master=Panel_parametros, textvariable=Frecuancia_Muestreo,width = 10,relief='solid').grid(row=4, column=1)
Factor_Ruido = tk.IntVar()
tb_FR = tk.Entry(master=Panel_parametros, textvariable=Factor_Ruido,width = 10,relief='solid').grid(row=5, column=1)

btn_Modificar = tk.Button(master=Panel_parametros, bg='#c75050',fg='white',relief='flat',text='Cambiar parametros',width=10, command = cambiar_parametros).grid(row=6, column=0)

"""    Modificacion de parametros                                                                                    """



lbl_titulo1 = tk.Label(master=Panel_parametros, bg = 'white',fg = '#ffa72b',font=('Times', 15, 'bold italic'), text="                                                ").grid(row=7, column=0,columnspan = 7, rowspan = 7 )



#Creación del panel de soluciones de ecuaciones diferenciales-----------------------------------------------------------
Panel_Metodos_Solucion = tk.Frame(master = window) #Se asocia a la ventana principal
Panel_Metodos_Solucion.place(x=480, y=350)
Panel_Metodos_Solucion.config(bg="#FFFFFF", width=300, height=300, relief=tk.SOLID, bd=3)

#Titulo del panel de metodos
lbl_titulo1 = tk.Label(master=Panel_Metodos_Solucion, bg = 'white',fg = '#6ca89f',font=('Times', 15, 'bold italic'), text="                                       ").grid(row=0, column=0,columnspan = 2, rowspan = 1 )
lbl_titulo2 = tk.Label(master=Panel_Metodos_Solucion, bg = 'white',fg = '#6ca89f',font=('Times', 15, 'bold italic'), text="          Método de solución ED        ").grid(row=1, column=0,columnspan = 2, rowspan = 1 )

#Labels  del chechbox
l_eulerFor = tk.Label(master= Panel_Metodos_Solucion,bg = '#265834', fg = 'white',text="Euler adelante",relief='groove',width = 15).grid(row=2, column=1)
l_eulerFor = tk.Label(master= Panel_Metodos_Solucion,bg = '#265834', fg = 'white',text="Euler atras",relief='groove',width = 15).grid(row=3, column=1)
l_eulerFor = tk.Label(master= Panel_Metodos_Solucion,bg = '#265834', fg = 'white',text="Euler modificado",relief='groove',width = 15).grid(row=4, column=1)
l_eulerFor = tk.Label(master= Panel_Metodos_Solucion,bg = '#265834', fg = 'white',text="Runge Kutta 2",relief='groove',width = 15).grid(row=5, column=1)
l_eulerFor = tk.Label(master= Panel_Metodos_Solucion,bg = '#265834', fg = 'white',text="Runge Kutta 4",relief='groove',width = 15).grid(row=6, column=1)

"""

#Checkboxes-----------------------------------------------------------------------------------------------------------
"""
val_eulerFor = tk.IntVar()
check_eulerfor = tk.Checkbutton(master = Panel_Metodos_Solucion, text="",command = Graficador ,variable=val_eulerFor, onvalue=1, offvalue=0).grid(row=2, column=0)
val_eulerBack = tk.IntVar()
check_eulerBack = tk.Checkbutton(master = Panel_Metodos_Solucion, text="",command = Graficador,variable=val_eulerBack, onvalue=1, offvalue=0).grid(row=3, column=0)
val_eulerMod = tk.IntVar()
check_eulerMod = tk.Checkbutton(master = Panel_Metodos_Solucion, text="",command = Graficador,variable=val_eulerMod, onvalue=1, offvalue=0).grid(row=4, column=0)
val_Ruggen2 = tk.IntVar()
check_Ruggen2 = tk.Checkbutton(master = Panel_Metodos_Solucion, text="",command = Graficador,variable=val_Ruggen2, onvalue=1, offvalue=0).grid(row=5, column=0)
val_Ruggen4 = tk.IntVar()
check_Ruggen4 = tk.Checkbutton(master = Panel_Metodos_Solucion, text="",command = Graficador,variable=val_Ruggen4, onvalue=1, offvalue=0).grid(row=6, column=0)
#Boton de cerrar ventana funcion---------------------------------------------------------------------------------------------
def CerrarAplicacion():
    MsgBox = tk.messagebox.askquestion ('Cerrar Aplicación','¿Está seguro que desea cerrar la aplicación?',icon = 'warning')
    if MsgBox == 'yes':
       window.destroy()
    else:
        tk.messagebox.showinfo('Retornar','Será retornado a la aplicación')

#Creación de boton para cerrar ventana
btn_Cancelar = tk.Button(window, bg='#c75050',fg='white',relief='flat',text='X',width=10, command = CerrarAplicacion).place(x=0,y=0)
#Boton cargar datos------------------------------------------------------------------------------------------------------------


btn_CargarDatos = tk.Button(window, bg='#54a5bd',fg='white',relief='solid',text='Cargar Datos',width=12, command = Cargar).place(x=100,y=0)
#Boton exportar datos----------------------------------------------------------------------------------------------------------

btn_ExportarDatos = tk.Button(window, bg='#54a5bd',fg='white',relief='solid',text='Exportar datos',width=12, command =Exportar).place(x=201,y=0)


#Imagen corazaon
image = Image.open("corazon.png")
image = image.resize((80, 80), Image.ANTIALIAS)
img= ImageTk.PhotoImage(image)
lab=tk.Label(image=img)
lab.place(x=15,y=40)

#Label central de título
tituloCentral_ECG = tk.Label(master=window, bg="#fed264", font=('Comic Sans MS', 15, 'bold italic'), text=f"Señal de ECG", relief='solid',width=20).place(x=140,y=80)




#Boton  HR
btn_HR = tk.Button(window, bg='#0b6c90',fg='white',relief='solid',text='HR',width=12, command = Heart_Rate).place(x=140,y=450)
#Campo de texto

#Label del resultado de la funcion de HR
resultado_HR = tk.StringVar()
lbl_result = tk.Label(master=window, fg = 'black', bg = 'white',relief='solid', textvariable=resultado_HR, text="", width =17).place(x=280, y=450)



#Label editar
btn_EditarAB = tk.Button(window, bg='white',fg='#8ec44c',relief='solid',text="Cambiar\n los puntos",width=12, command = cambiarAB).place(x=35,y=550)
#lbl_result2 = tk.Label(master=window, fg = '#8ec44c',  text="Cambiar\n los puntos").place(x=35, y=550)

#Edicion de parametros
image2 = Image.open("lapiz.png")
image2 = image2.resize((40, 40))
img2= ImageTk.PhotoImage(image2)
lab=tk.Label(image=img2)
lab.place(x=40,y=600)

#Grilla de edicion de parametros
grilla_parametros = tk.Frame(master = window) #Se asocia a la ventana principal
grilla_parametros.place(x=140, y=530)
grilla_parametros.config(bg="white", width=310, height=120, relief='flat', bd=3)

#Label titulo
lbl_void = tk.Label(master=grilla_parametros, fg = 'white',bg = '#8ec44c', text="            ").grid(row=0, column=0)
lbl_P = tk.Label(master=grilla_parametros, fg = 'white',bg = '#8ec44c', text="     P       ").grid(row=0, column=1)
lbl_Q = tk.Label(master=grilla_parametros, fg = 'white',bg = '#8ec44c', text="     Q       ").grid(row=0, column=2)
lbl_R = tk.Label(master=grilla_parametros, fg = 'white',bg = '#8ec44c', text="     R       ").grid(row=0, column=3)
lbl_S = tk.Label(master=grilla_parametros, fg = 'white',bg = '#8ec44c', text="     S       ").grid(row=0, column=4)
lbl_T = tk.Label(master=grilla_parametros, fg = 'white',bg = '#8ec44c', text="     T       ").grid(row=0, column=5)

#aI Y BI
lbl_ai =  tk.Label(master=grilla_parametros, fg = 'black', bg = 'white',font=('Comic Sans MS', 15, 'bold italic'), relief='flat',  text="ai", ).grid(row=1, column=0)
lbl_bi =  tk.Label(master=grilla_parametros, fg = 'black', bg = 'white',font=('Comic Sans MS', 15, 'bold italic'), relief='flat',  text="bi", ).grid(row=2, column=0)

#Text boxes AI
Pa = tk.IntVar()
tb_Pa = tk.Entry(master=grilla_parametros, textvariable=Pa,width = 4).grid(row=1, column=1)
Qa =tk.IntVar()
tb_Qa = tk.Entry(master=grilla_parametros, textvariable=Qa,width = 4).grid(row=1, column=2)
Ra =tk.IntVar()
tb_Ra = tk.Entry(master=grilla_parametros, textvariable=Ra,width = 4).grid(row=1, column=3)
Sa =tk.IntVar()
tb_Sa = tk.Entry(master=grilla_parametros, textvariable=Sa,width = 4).grid(row=1, column=4)
Ta =tk.IntVar()
tb_Ta = tk.Entry(master=grilla_parametros, textvariable=Ta,width = 4).grid(row=1, column=5)

#Text boxes bi
Pb =tk.IntVar()
tb_Pb = tk.Entry(master=grilla_parametros, textvariable=Pb,width = 4).grid(row=2, column=1)
Qb =tk.IntVar()
tb_Qb= tk.Entry(master=grilla_parametros, textvariable=Qb,width = 4).grid(row=2, column=2)
Rb =tk.IntVar()
tb_Rb = tk.Entry(master=grilla_parametros, textvariable=Rb,width = 4).grid(row=2, column=3)
Sb =tk.IntVar()
tb_Sb = tk.Entry(master=grilla_parametros, textvariable=Sb,width = 4).grid(row=2, column=4)
Tb =tk.IntVar()
tb_Tb = tk.Entry(master=grilla_parametros, textvariable=Tb,width = 4).grid(row=2, column=5)


window.mainloop()



