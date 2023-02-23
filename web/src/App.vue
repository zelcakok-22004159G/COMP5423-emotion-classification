<template lang="pug">
#app
  .main.box.centered.has-text-centered
    .flex-box
      label.subtitle COMP5423 Homework 1 ( Kok Tsz Ho 22004159G )
      .emoji(:class="{ scaleOut: change, scaleIn: !change }")
        label.title Guess the emotion
        br
        br
        label.subtitle {{ display }}
      input.input(
        v-model="text", 
        placeholder="Type something here...", 
        @input="think"
      )
</template>

<script>
import axios from "axios"

const ENDPOINT = "http://localhost:5000/api"

const EMOTIONS = {
  "anger": "Anger",
  "fear": "Fear",
  "joy": "Joy",
  "love": "Love",
  "sadness": "Sadness",
  "surprise": "Surprise",
}

const EMOJIS = {
  "anger": "😡",
  "fear": "😨",
  "joy": "😆",
  "love": "😍",
  "sadness": "😢",
  "surprise": "🤩",
}

export default {
  name: 'App',
  data() {
    return {
      text: null,
      emotion: null,
      task: null,
      change: false
    }
  },
  methods: {
    think() {
      this.emotion = `Thinking`
      if (this.task) {
        clearTimeout(this.task)
      }
      this.task = setTimeout(async () => {
        this.change = true;
        const { data } = await axios.get(ENDPOINT + `/classify?q=${this.text}`)
        if (!data || !data?.result) {
          this.emotion = "I don't know..."
        } else {
          this.emotion = `${data.result}`
        }
        this.change = false;
      }, 500)
    }
  },
  computed: {
    display() {
      if (this.emotion === `Thinking`) {
        return `Thinking 🤔`
      }
      else if (!this.text || !this.emotion) {
        const values = Object.values(EMOTIONS)
        const emotions = values.map(val=>val = `${val} ${EMOJIS[val.toLowerCase()]}`)
        return `[ ${emotions.join(" | ")} ]`;
      } else {
        return `${EMOTIONS[this.emotion]} ${EMOJIS[this.emotion]}`;
      }
    }
  }
}
</script>

<style lang="scss">
@import "../node_modules/bulma/css/bulma.css";

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 160px;

  .main {
    max-width: 700px;
    height: 300px;

    .flex-box {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      align-items: center;

      .emoji {
        margin-bottom: 32px;
      }
    }
  }

  .centered {
    margin: auto;
  }
}

.scaleOut {
  animation: scale-out 1s forwards;
}

@keyframes scale-out {
  0% {
    scale: 100%;
  }

  25% {
    scale: 0.75%;
  }

  50% {
    scale: 0.5%;
  }

  75% {
    scale: 0.25%;
  }

  100% {
    scale: 0;
  }
}

.scaleIn {
  animation: scale-in 0.5s forwards;
}

@keyframes scale-in {
  0% {
    scale: 0;
  }

  25% {
    scale: 0.25%;
  }

  50% {
    scale: 0.5%;
  }

  75% {
    scale: 0.75%;
  }

  100% {
    scale: 100%;
  }
}
</style>
