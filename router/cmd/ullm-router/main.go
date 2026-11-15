// Command ullm-router is the Go router and gateway in front of the uLLM engine
// replicas: prefix-aware routing, health checks, retries, a circuit breaker and
// telemetry fan-in. It is built in milestone M5 (docs/AGENT_ROADMAP.md, 6.7).
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "ullm-router: not built yet (milestone M5)")
	os.Exit(1)
}
