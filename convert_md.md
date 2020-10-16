
```python
# 1st layer
cnv1 = self.conv1(x)
lrn1 = self.lrn1(cnv1)
mp1 = self.pool1(lrn1)

# 2nd layer
cnv2 = self.conv2(mp1)
lrn2 = self.lrn2(cnv2)
mp2 = self.pool2(lrn2)

# 3rd layer
cnv3 = self.conv3(mp2)

# 4th layer
cnv4 = self.conv4(cnv3)

# 5th layer
cnv5 = self.conv5(cnv4)
lrn3 = self.lrn3(cnv5)
mp3 = self.pool3(cnv5)

ft = self.flatten(mp3)

fcl1 = self.fc1(ft)
if training:
    do1 = self.dropout1(fcl1, training= training)
    fcl2 = self.fc2(do1)
    do2 = self.dropout2(fcl2, training= training)
    fcl3 = self.fc3(do2)

else:
    # multiply their outputs by 0.5
    mul1 = self.mul1(fcl1)
    fcl2 = self.fc2(mul1)
    mul2 = self.mul2(fcl2)
    fcl3 = self.fc3(mul2)
```