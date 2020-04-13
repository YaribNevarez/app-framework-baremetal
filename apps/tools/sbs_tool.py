#!/usr/bin/env python
# SBS Tool
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import struct
import serial

SerialPort_frameSignature = 0x5A
CMD_CLEAR         = 0x00
CMD_PLOT          = 0x01
CMD_SET_VISIBLE   = 0x02
CMD_SET_STEP_TIME = 0x03
CMD_SET_TIME      = 0x04
CMD_TEXT_MSG      = 0x05
CMD_BYTE_BUFFER   = 0x06

if __name__ == '__main__':
    fig, axs = plt.subplots(2)
    plt.ion()

    
    fig.suptitle('SbS platform')

    port = '/dev/ttyUSB1'
    serial_port = serial.Serial(port=port, baudrate=115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
    
    while True:
        if serial_port.readable():
            frameSignature = int.from_bytes(serial_port.read(), byteorder='big', signed=False)

            if frameSignature == SerialPort_frameSignature:
                frameSize = int.from_bytes(serial_port.read(), byteorder='big', signed=False)
                command = int.from_bytes(serial_port.read(), byteorder='big', signed=False)

                if command == CMD_CLEAR:
                    pass
                elif command == CMD_PLOT:
                    trace = int.from_bytes(serial_port.read(), byteorder='big', signed=False)
                    length = int.from_bytes(serial_port.read(), byteorder='big', signed=False)
                    
                    plot_x = np.array([])
                    plot_y = np.array([])
                    Y = np.array([1])
                    X = np.array([1])

                    for i in range(length // 2):
                        x = np.float64(struct.unpack('d', serial_port.read(8))[0])
                        plot_x = np.append(plot_x, x)

                        y = np.float64(struct.unpack('d', serial_port.read(8))[0])
                        plot_y = np.append(plot_y, y)
                    crc = int.from_bytes(serial_port.read(), byteorder='big', signed=False)
                    
                    axs[trace].clear()
                    axs[trace].plot(plot_x, plot_y)

                    print (plot_x[2] - plot_x[1]);

                    plt.show()
                elif command == CMD_SET_VISIBLE:
                    pass
                if command == CMD_SET_STEP_TIME:
                    pass
                elif command == CMD_SET_TIME:
                    pass
                elif command == CMD_BYTE_BUFFER:
                    bufferSize = int.from_bytes(serial_port.read(), byteorder='big', signed=False)
                    
                    plot_y = np.array([])
                    plot_x = np.array([])

                    x = np.array([1])
                    y = np.array([1])
                    
                    for i in range(bufferSize // 4): # Size of the data
                        y = struct.unpack('>f', serial_port.read(size=4))[0]
                        plot_y = np.append(plot_y, y)
                        x += 1
                        plot_x = np.append(plot_x, x)

                    crc = int.from_bytes(serial_port.read(), byteorder='big', signed=False)

                    axs[1].clear()
                    axs[1].step(plot_x, plot_y, where='mid', label='mid')

                    plt.show()

                elif command == CMD_TEXT_MSG:
                    id = int.from_bytes(serial_port.read(), byteorder='big', signed=False)
                    msgSize = int.from_bytes(serial_port.read(), byteorder='big', signed=False)
                    msg = serial_port.read(msgSize)
                    print(msg)
                    crc = int.from_bytes(serial_port.read(), byteorder='big', signed=False)
        plt.pause(0.01)