package plan

type TaskSpec struct {
	DeviceID       int
	NetAddr        NetAddr
	MonitoringPort uint16

	GlobalRank int // FIXME: make it dynamic

	SockFile string
}
