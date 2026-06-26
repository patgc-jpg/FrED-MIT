######### Program to control the FrED's DC motor ########
#
# Two modes (set MODE below):
#   MODE = "identify" -> OPEN LOOP: apply a known input profile and log speed(t).
#                        Use this data to fit the system model.
#   MODE = "control"  -> CLOSED LOOP: chosen controller drives motor to rpm_reference.

import os
import cv2
import time
import math
import board
import busio
import atexit
import datetime
import digitalio
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO
import adafruit_mcp3xxx.mcp3008 as MCP
GPIO.setmode(GPIO.BCM)
import FrED_functions

from time import sleep
import spidev
from adafruit_mcp3xxx.analog_in import AnalogIn


#######################################################
########## CONFIG: set these before running ###########
#######################################################
MODE       = "identify"   # "identify" (open loop, for system ID) or "control" (closed loop)
PROFILE    = 1            # open-loop profile, used ONLY when MODE == "identify"
                          #   1 -> constant step  (motor_input = 29)
                          #   2 -> least-squares / PRBS excitation
CONTROLLER = "PID"        # controller to use, ONLY when MODE == "control"
                          #   "PID" | "PI" | "STSM"

rpm_reference    = 45     # desired motor speed (rpm), used ONLY in "control" mode
opcion_LS        = 11     # least-squares option,      used ONLY when PROFILE == 2
PROFILE_DURATION = 30     # s — duration of the identify sequence; alert prints 5 s after this

########## sample / timing ##########
tm         = 0.02
match_time = 0.020


########## GPIO Pin Definitions ##########
SLAVE_SELECT_ENC = 1
motorPin         = 5

########## Initialise GPIO ##########
GPIO.setwarnings(False)
GPIO.setup(motorPin, GPIO.OUT)

########## Create the SPI bus / ADC ##########
spi       = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
cs        = digitalio.DigitalInOut(board.D8)
mcp       = MCP.MCP3008(spi, cs)
channel_0 = AnalogIn(mcp, MCP.P0)

########## DC Motor initialisation ##########
ppr      = 300.8
dcFreq   = 2500
fanFreq  = 1000

oldtime      = 0
oldpos       = 0
lasttime     = 0
motor_output = GPIO.PWM(motorPin, dcFreq)
motor_output.start(0)
frame_count  = 0

GPIO.setup(SLAVE_SELECT_ENC, GPIO.OUT)
GPIO.output(SLAVE_SELECT_ENC, GPIO.HIGH)


#######################################################
########## Encoder functions ##########################
#######################################################
def initialize_encoder():
    global spi_enc
    spi_enc = spidev.SpiDev()
    spi_enc.open(0, 0)
    spi_enc.max_speed_hz = 50000
    GPIO.output(SLAVE_SELECT_ENC, GPIO.LOW)
    spi_enc.xfer2([0x88, 0x03])
    GPIO.output(SLAVE_SELECT_ENC, GPIO.HIGH)
    clear_encoder_count()


def clear_encoder_count():
    GPIO.output(SLAVE_SELECT_ENC, GPIO.LOW)
    spi_enc.xfer2([0x98, 0x00, 0x00, 0x00, 0x00])
    GPIO.output(SLAVE_SELECT_ENC, GPIO.HIGH)
    time.sleep(0.0001)
    GPIO.output(SLAVE_SELECT_ENC, GPIO.LOW)
    spi_enc.xfer2([0xE0])
    GPIO.output(SLAVE_SELECT_ENC, GPIO.HIGH)


def read_encoder():
    GPIO.output(SLAVE_SELECT_ENC, GPIO.LOW)
    spi_enc.xfer2([0x60])
    count_1 = spi_enc.xfer2([0x00])
    count_2 = spi_enc.xfer2([0x00])
    count_3 = spi_enc.xfer2([0x00])
    count_4 = spi_enc.xfer2([0x00])
    GPIO.output(SLAVE_SELECT_ENC, GPIO.HIGH)
    return (
        (count_1[0] << 24) +
        (count_2[0] << 16) +
        (count_3[0] << 8) +
        count_4[0]
    )


#######################################################
########## Open-loop input profiles (for ID) ##########
#######################################################
def motor_input_profile(current_time, option):
    """Return motor input for open-loop system identification."""
    if option == 1:
        # Constant step — simplest excitation for a first-order fit
        output = 29
    elif option == 2:
        # Least-squares / PRBS excitation — richer signal for ARX / LS models
        output = FrED_functions.least_square(current_time, opcion_LS)
    else:
        output = 0
    return output


#######################################################
########## Controller state ###########################
#######################################################
previous_time     = 0
previous_PIDtime  = 0
previous_steps    = 0
previous_rpm      = 0
previous_PIDerror = 0
PWM_motor            = 0
error_sum            = 0
u                    = 0
muestra              = 1
sequence_alert_shown = False

########## Data lists ##########
time_data          = []
rpm_data           = []
rpm_raw_data       = []
rpm_ref_data       = []
motor_input_data   = []
PWM_motor_data     = []
motor_voltage_data = []

########## Live plot ##########
plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot(time_data, rpm_data,     label='Motor speed (rpm)')
line2, = ax.plot(time_data, rpm_ref_data, label='rpm reference (rpm)')
ax.legend()
ax.set_title('DC Motor - ' + MODE)


def ploting():
    line1.set_xdata(time_data); line1.set_ydata(rpm_data)
    line2.set_xdata(time_data); line2.set_ydata(rpm_ref_data)
    ax.relim(); ax.autoscale_view(); plt.draw(); plt.pause(0.01)


def plotD():
    plt.plot(time_data, rpm_data)
    plt.show()


#######################################################
########## Main loop ##################################
#######################################################
tstart       = time.perf_counter()
initial_time = time.time()

initialize_encoder()
previous_position = 4294967269
first_sample      = True

try:
    while True:
        current_time = time.perf_counter() - tstart

        current_position = read_encoder()
        if first_sample:
            current_position = 4294967265
            first_sample     = False

        # 1) measure speed
        rpm_raw           = FrED_functions.motor_speed(current_time, previous_time,
                                                       previous_position, current_position)
        previous_time     = current_time
        previous_position = current_position

        rpm          = FrED_functions.filter(rpm_raw, previous_rpm)
        previous_rpm = rpm

        # 2) decide motor input
        if MODE == "control":
            if CONTROLLER == "PID":
                motor_input, error_i, PIDerror = FrED_functions.PID(
                    rpm_reference, rpm, previous_PIDerror,
                    error_sum, current_time, previous_PIDtime)
                previous_PIDerror = PIDerror
            elif CONTROLLER == "PI":
                motor_input, error_i = FrED_functions.PI(
                    rpm_reference, rpm, error_sum,
                    current_time, previous_PIDtime)
                previous_PIDerror = 0
            elif CONTROLLER == "STSM":
                motor_input, error_i = FrED_functions.STSM(
                    rpm_reference, rpm, u, error_sum,
                    current_time, previous_PIDtime)
                previous_PIDerror = 0
            previous_PIDtime = current_time
            error_sum        = error_i
            ref_now          = rpm_reference
        else:  # "identify" -> open loop, ignore the controller
            motor_input = motor_input_profile(current_time, PROFILE)
            ref_now     = 0
            if not sequence_alert_shown and current_time >= PROFILE_DURATION + 5:
                print("\n*** SECUENCIA TERMINADA ***\n")
                sequence_alert_shown = True

        # 3) apply linearization and clamp
        PWM_motor = FrED_functions.linearization(motor_input)
        PWM_motor = max(min(PWM_motor, 100), 0)
        motor_output.ChangeDutyCycle(PWM_motor)

        # 4) console feedback (time \t rpm \t motor_input)
        if current_time >= muestra:
            print(f"{current_time}\t{rpm}\t{motor_input}")
            muestra += 1

        # 5) store data
        motor_voltage = (12 * PWM_motor) / 100

        time_data.append(round(current_time, 2))
        rpm_data.append(round(rpm, 2))
        rpm_raw_data.append(round(rpm_raw, 2))
        rpm_ref_data.append(ref_now)
        motor_input_data.append(round(motor_input, 2))
        PWM_motor_data.append(round(PWM_motor, 2))
        motor_voltage_data.append(round(motor_voltage, 2))

        # 6) keep a consistent sample time
        wait = max(0, match_time - current_time)
        time.sleep(wait)
        match_time += tm

except KeyboardInterrupt:
    print("\nCode Stopped\n")
    FrED_functions.save_data(time_data, rpm_data, motor_voltage_data,
                             motor_input_data, PWM_motor_data, rpm_raw_data)
    print("Data saved in FrED_data.txt file\n\n")

finally:
    GPIO.cleanup()
