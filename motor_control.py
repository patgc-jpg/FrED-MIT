######### Program to control the FrED's DC motor ########
#
# Two modes, chosen from the TERMINAL (see examples below):
#   --mode identify -> OPEN LOOP: apply a known input profile and log speed(t).
#                      Use this data to fit the system model.
#   --mode control  -> CLOSED LOOP: chosen controller drives motor to --setpoint
#                      using --kp/--ki/--kd (only when --controller PID).
#
# Examples:
#   python motor_control.py --mode identify --profile 1
#   python motor_control.py --mode identify --profile 2 --opcion_ls 11
#   python motor_control.py --mode control  --controller PID  --setpoint 45 --kp 0.3106 --ki 9.703 --kd 0.02044
#   python motor_control.py --mode control  --controller PI   --setpoint 45
#   python motor_control.py --mode control  --controller STSM --setpoint 45

import os
import cv2
import time
import math
import argparse
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
########## CONFIG: set via terminal arguments #########
#######################################################
parser = argparse.ArgumentParser(
    description="FrED DC motor: identify (open loop, for system ID) or control (closed loop).")
parser.add_argument("--mode", choices=["identify", "control"], default="identify",
                    help="identify = open loop for system ID; control = closed loop. Default: identify")
parser.add_argument("--profile", type=int, default=1,
                    help="open-loop test profile (1 or 2). Only used in identify mode. Default: 1")
parser.add_argument("--controller", choices=["PID", "PI", "STSM"], default="PID",
                    help="controller type. Only used in control mode. Default: PID")
parser.add_argument("--setpoint", type=float, default=45.0,
                    help="rpm setpoint. Only used in control mode. Default: 45")
parser.add_argument("--kp", type=float, default=0.3106,
                    help="PID proportional gain. Only used when controller=PID. Default: 0.3106")
parser.add_argument("--ki", type=float, default=9.703,
                    help="PID integral gain. Only used when controller=PID. Default: 9.703")
parser.add_argument("--kd", type=float, default=0.02044,
                    help="PID derivative gain. Only used when controller=PID. Default: 0.02044")
parser.add_argument("--opcion_ls", type=int, default=11,
                    help="least-squares option for profile 2. Default: 11")
parser.add_argument("--duration", type=float, default=30.0,
                    help="identify sequence duration in seconds. Default: 30")
args = parser.parse_args()

MODE             = args.mode
PROFILE          = args.profile
CONTROLLER       = args.controller
rpm_reference    = args.setpoint
opcion_LS        = args.opcion_ls
PROFILE_DURATION = args.duration
KP, KI, KD      = args.kp, args.ki, args.kd

# Human-readable descriptions (used in the banner)
PROFILE_DESC = {
    1: "constant step, open loop",
    2: "rich PRBS excitation for least squares",
}
CONTROLLER_DESC = {
    "PID":  "classic three-term proportional-integral-derivative",
    "PI":   "proportional-integral, no derivative term",
    "STSM": "robust Super-Twisting sliding mode control",
}

########## safety / limits ##########
MOTOR_PWM_CEILING = 100   # % — no hay limite explicito de RPM en este codigo

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
dcFreq   = 1000
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
########## PID step (gains settable from terminal) ####
#######################################################
def pid_step(reference, measurement, prev_error, error_sum,
             current_time, prev_time, kp, ki, kd):
    """One PID iteration with gains passed as arguments."""
    delta_time = current_time - prev_time
    error   = reference - measurement
    error_d = (error - prev_error) / delta_time if delta_time > 0 else 0.0
    error_i = (error * delta_time) + error_sum
    output  = kp * error + ki * error_i + kd * error_d
    return output, error_i, error


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
########## Startup banner #############################
#######################################################
print("=" * 60)
print(" FrED MOTOR DC")
print("=" * 60)
if MODE == "identify":
    desc = PROFILE_DESC.get(PROFILE, "UNKNOWN profile -> motor stays off")
    print(f" Mode       : IDENTIFY  (open loop, system identification)")
    print(f" Profile    : {PROFILE}  ->  {desc}")
    print(f" Duration   : {PROFILE_DURATION:.0f} s")
else:  # control
    ctrl_desc = CONTROLLER_DESC.get(CONTROLLER, "UNKNOWN controller")
    print(f" Mode       : CONTROL  (closed loop)")
    print(f" Controller : {CONTROLLER}  ->  {ctrl_desc}")
    print(f" Setpoint   : {rpm_reference:.1f} rpm")
    if CONTROLLER == "PID":
        print(f" Gains      : Kp = {KP}   Ki = {KI}   Kd = {KD}")
    else:
        print(f" Gains      : fixed inside FrED_functions")
print(f" PWM max    : {MOTOR_PWM_CEILING}%  (no explicit RPM ceiling in this code)")
print(f" Sample     : {tm * 1000:.1f} ms")
print("=" * 60)
print(" Press Ctrl+C to stop and save data to FrED_data.txt")
print("=" * 60)
print("  time(s)\trpm\tmotor_input")

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
                motor_input, error_i, PIDerror = pid_step(
                    rpm_reference, rpm, previous_PIDerror,
                    error_sum, current_time, previous_PIDtime, KP, KI, KD)
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
        PWM_motor = max(min(PWM_motor, MOTOR_PWM_CEILING), 0)
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
