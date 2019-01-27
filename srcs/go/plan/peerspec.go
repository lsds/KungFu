package plan

type PeerSpec struct {
	DeviceID       int
	NetAddr        NetAddr
	MonitoringPort uint16
	GlobalRank     int
}
