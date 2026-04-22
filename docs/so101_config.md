# SO101 config

## MotorsBus(Port)

* Follower arm: `/dev/tty.usbmodem5B141136491`
* Leader arm: `/dev/tty.usbmodem5B141123031`


## Motor reset commands

### Follower
```bash
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B141136491
```

### Leader
```bash
lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5B141123031
```


## Motor Calibration commands

### Follower
```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B141136491
    --robot.id=follower
```

### Leader
```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5B141123031
    --teleop.id=leader
```
