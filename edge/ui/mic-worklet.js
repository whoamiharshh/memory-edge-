"use strict";
// AudioWorklet: copies raw microphone samples (mono, the context's native rate, e.g. 48 kHz) to the page in blocks.
// No processing here: the device does the physics on the raw sound.
class MicTap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("mic-tap", MicTap);
