import time
import lgpio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# Define the I2C address and bus number
I2C_BUS = 1
I2C_ADDRESS = 0x40

# Initialize the I2C connection using lgpio
try:
    i2c = lgpio.i2c_open(I2C_BUS, I2C_ADDRESS)
    print(f"Successfully opened I2C connection to address 0x{I2C_ADDRESS:X}")
except Exception as e:
    print(f"Failed to open I2C connection: {e}")
    exit(1)

# Create a PCA9685 instance
try:
    pca = PCA9685(i2c)
    pca.frequency = 50  # Standard frequency for servos is 50 Hz
except Exception as e:
    print(f"Failed to initialize PCA9685: {e}")
    lgpio.i2c_close(i2c)
    exit(1)

# Create a servo object for a specific channel (e.g., channel 0)
servo0 = servo.Servo(pca.channels[0])

# Function to set the angle of the servo
def set_servo_angle(channel, angle):
    servo_channel = servo.Servo(pca.channels[channel])
    servo_channel.angle = angle

# Main loop to test the servo
try:
    while True:
        for angle in range(0, 180, 5):  # 0 to 180 degrees, step by 5 degrees
            set_servo_angle(0, angle)
            time.sleep(0.1)
        for angle in range(180, 0, -5):  # 180 to 0 degrees, step by -5 degrees
            set_servo_angle(0, angle)
            time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    # Cleanup
    lgpio.i2c_close(i2c)
    pca.deinit()
