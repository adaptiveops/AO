import lgpio

I2C_BUS = 1
I2C_ADDRESS = 0x40

try:
    h = lgpio.i2c_open(I2C_BUS, I2C_ADDRESS)
    print(f"Successfully opened I2C connection to address 0x{I2C_ADDRESS:X}")
    lgpio.i2c_close(h)
except Exception as e:
    print(f"Failed to open I2C connection: {e}")
