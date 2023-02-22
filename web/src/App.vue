<template lang="pug">
#app
  .main.box.centered.has-text-centered
    .flex-box
      label.subtitle COMP5423 Homework 1 ( Kok Tsz Ho 22004159G )
      .emoji(:class="{ scaleOut: false, scaleIn: true }")
        label.title {{ display }}
      input.input(
        v-model="text", 
        placeholder="Type something here...", 
        @input="think"
      )
</template>

<script>
import axios from "axios"

const ENDPOINT = "http://localhost:5000/api"

export default {
  name: 'App',
  data() {
    return {
      text: null,
      emotion: null,
      task: null
    }
  },
  methods: {
    think() {
      if (this.task) {
        clearTimeout(this.task)
      }
      this.emotion = "Thinking..."
      this.task = setTimeout(async () => {
        console.log("Inputed", this.text);
        const { data } = await axios.get(ENDPOINT + `/classify?q=${this.text}`)
        if (!data || !data?.result) {
          this.emotion = "I don't know..."
        } else {
          this.emotion = `${data.result}`
        }
      }, 500)
    }
  },
  computed: {
    display() {
      if (!this.emotion) {
        return "Waiting for text..."
      } else {
        return this.emotion.toUpperCase()
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
    max-width: 600px;
    height: 200px;

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
  animation: scale-out 2s forwards;
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
