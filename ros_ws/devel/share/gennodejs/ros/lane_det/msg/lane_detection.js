// Auto-generated. Do not edit!

// (in-package lane_det.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class lane_detection {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.left_1 = null;
      this.left_0 = null;
      this.right_0 = null;
      this.right_1 = null;
      this.coeff_left_1 = null;
      this.coeff_left_0 = null;
      this.coeff_right_0 = null;
      this.coeff_right_1 = null;
    }
    else {
      if (initObj.hasOwnProperty('left_1')) {
        this.left_1 = initObj.left_1
      }
      else {
        this.left_1 = [];
      }
      if (initObj.hasOwnProperty('left_0')) {
        this.left_0 = initObj.left_0
      }
      else {
        this.left_0 = [];
      }
      if (initObj.hasOwnProperty('right_0')) {
        this.right_0 = initObj.right_0
      }
      else {
        this.right_0 = [];
      }
      if (initObj.hasOwnProperty('right_1')) {
        this.right_1 = initObj.right_1
      }
      else {
        this.right_1 = [];
      }
      if (initObj.hasOwnProperty('coeff_left_1')) {
        this.coeff_left_1 = initObj.coeff_left_1
      }
      else {
        this.coeff_left_1 = [];
      }
      if (initObj.hasOwnProperty('coeff_left_0')) {
        this.coeff_left_0 = initObj.coeff_left_0
      }
      else {
        this.coeff_left_0 = [];
      }
      if (initObj.hasOwnProperty('coeff_right_0')) {
        this.coeff_right_0 = initObj.coeff_right_0
      }
      else {
        this.coeff_right_0 = [];
      }
      if (initObj.hasOwnProperty('coeff_right_1')) {
        this.coeff_right_1 = initObj.coeff_right_1
      }
      else {
        this.coeff_right_1 = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type lane_detection
    // Serialize message field [left_1]
    // Serialize the length for message field [left_1]
    bufferOffset = _serializer.uint32(obj.left_1.length, buffer, bufferOffset);
    obj.left_1.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Point.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [left_0]
    // Serialize the length for message field [left_0]
    bufferOffset = _serializer.uint32(obj.left_0.length, buffer, bufferOffset);
    obj.left_0.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Point.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [right_0]
    // Serialize the length for message field [right_0]
    bufferOffset = _serializer.uint32(obj.right_0.length, buffer, bufferOffset);
    obj.right_0.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Point.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [right_1]
    // Serialize the length for message field [right_1]
    bufferOffset = _serializer.uint32(obj.right_1.length, buffer, bufferOffset);
    obj.right_1.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Point.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [coeff_left_1]
    bufferOffset = _arraySerializer.float32(obj.coeff_left_1, buffer, bufferOffset, null);
    // Serialize message field [coeff_left_0]
    bufferOffset = _arraySerializer.float32(obj.coeff_left_0, buffer, bufferOffset, null);
    // Serialize message field [coeff_right_0]
    bufferOffset = _arraySerializer.float32(obj.coeff_right_0, buffer, bufferOffset, null);
    // Serialize message field [coeff_right_1]
    bufferOffset = _arraySerializer.float32(obj.coeff_right_1, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type lane_detection
    let len;
    let data = new lane_detection(null);
    // Deserialize message field [left_1]
    // Deserialize array length for message field [left_1]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.left_1 = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.left_1[i] = geometry_msgs.msg.Point.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [left_0]
    // Deserialize array length for message field [left_0]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.left_0 = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.left_0[i] = geometry_msgs.msg.Point.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [right_0]
    // Deserialize array length for message field [right_0]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.right_0 = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.right_0[i] = geometry_msgs.msg.Point.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [right_1]
    // Deserialize array length for message field [right_1]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.right_1 = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.right_1[i] = geometry_msgs.msg.Point.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [coeff_left_1]
    data.coeff_left_1 = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [coeff_left_0]
    data.coeff_left_0 = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [coeff_right_0]
    data.coeff_right_0 = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [coeff_right_1]
    data.coeff_right_1 = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 24 * object.left_1.length;
    length += 24 * object.left_0.length;
    length += 24 * object.right_0.length;
    length += 24 * object.right_1.length;
    length += 4 * object.coeff_left_1.length;
    length += 4 * object.coeff_left_0.length;
    length += 4 * object.coeff_right_0.length;
    length += 4 * object.coeff_right_1.length;
    return length + 32;
  }

  static datatype() {
    // Returns string type for a message object
    return 'lane_det/lane_detection';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '6a8ccbd1be031a9ebcd2bcf06c6e28fd';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    geometry_msgs/Point[] left_1
    geometry_msgs/Point[] left_0
    geometry_msgs/Point[] right_0
    geometry_msgs/Point[] right_1
    float32[] coeff_left_1
    float32[] coeff_left_0
    float32[] coeff_right_0
    float32[] coeff_right_1
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new lane_detection(null);
    if (msg.left_1 !== undefined) {
      resolved.left_1 = new Array(msg.left_1.length);
      for (let i = 0; i < resolved.left_1.length; ++i) {
        resolved.left_1[i] = geometry_msgs.msg.Point.Resolve(msg.left_1[i]);
      }
    }
    else {
      resolved.left_1 = []
    }

    if (msg.left_0 !== undefined) {
      resolved.left_0 = new Array(msg.left_0.length);
      for (let i = 0; i < resolved.left_0.length; ++i) {
        resolved.left_0[i] = geometry_msgs.msg.Point.Resolve(msg.left_0[i]);
      }
    }
    else {
      resolved.left_0 = []
    }

    if (msg.right_0 !== undefined) {
      resolved.right_0 = new Array(msg.right_0.length);
      for (let i = 0; i < resolved.right_0.length; ++i) {
        resolved.right_0[i] = geometry_msgs.msg.Point.Resolve(msg.right_0[i]);
      }
    }
    else {
      resolved.right_0 = []
    }

    if (msg.right_1 !== undefined) {
      resolved.right_1 = new Array(msg.right_1.length);
      for (let i = 0; i < resolved.right_1.length; ++i) {
        resolved.right_1[i] = geometry_msgs.msg.Point.Resolve(msg.right_1[i]);
      }
    }
    else {
      resolved.right_1 = []
    }

    if (msg.coeff_left_1 !== undefined) {
      resolved.coeff_left_1 = msg.coeff_left_1;
    }
    else {
      resolved.coeff_left_1 = []
    }

    if (msg.coeff_left_0 !== undefined) {
      resolved.coeff_left_0 = msg.coeff_left_0;
    }
    else {
      resolved.coeff_left_0 = []
    }

    if (msg.coeff_right_0 !== undefined) {
      resolved.coeff_right_0 = msg.coeff_right_0;
    }
    else {
      resolved.coeff_right_0 = []
    }

    if (msg.coeff_right_1 !== undefined) {
      resolved.coeff_right_1 = msg.coeff_right_1;
    }
    else {
      resolved.coeff_right_1 = []
    }

    return resolved;
    }
};

module.exports = lane_detection;
