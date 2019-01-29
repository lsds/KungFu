package plan

// PeerSpec describes the system resources that will be used by the process of the peer
type PeerSpec struct {
	DeviceID       int
	NetAddr        NetAddr
	MonitoringPort uint16
}
